/*
 *  _reg_blockMatching_gpu.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright 2009 UCL - CMIC. All rights reserved.
 *
 */

#ifndef _REG_BLOCKMATCHING_GPU_CU
#define _REG_BLOCKMATCHING_GPU_CU

#include "blockMatchingKernel.h"

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"

#include <vector>
#include "_reg_maths.h"
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/*
*  before it was in the file _reg_blockMatching_kernels.cu
*
*
*  Created by Marc Modat and Pankaj Daga on 24/03/2009.
*  Copyright (c) 2009, University College London. All rights reserved.
*  Centre for Medical Image Computing (CMIC)
*  See the LICENSE.txt file in the nifty_reg root folder
*
*/
// Some parameters that we need for the kernel execution.
// The caller is supposed to ensure that the values are set

// Number of blocks in each dimension
__device__          __constant__ int3 c_BlockDim;
__device__ __constant__ int c_StepSize;
__device__          __constant__ uint3 c_ImageSize;
__device__ __constant__ float r1c1;

// Transformation matrix from nifti header
__device__          __constant__ float4 t_m_a;
__device__          __constant__ float4 t_m_b;
__device__          __constant__ float4 t_m_c;

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define OVERLAP_SIZE 3
#define STEP_SIZE 1


texture<float, 1, cudaReadModeElementType> targetImageArray_texture;
texture<float, 1, cudaReadModeElementType> resultImageArray_texture;
texture<int, 1, cudaReadModeElementType> totalBlock_texture;
/* *************************************************************** */
template<class DTYPE>
__inline__ __device__
void reg2D_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
   out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * 0 + (double)mat[0 * 4 + 3]);
   out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * 0 + (double)mat[1 * 4 + 3]);
   return;
}
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
   out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
   out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
   out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
   return;
}
// Apply the transformation matrix
__device__ inline void apply_affine(const float4 &pt, float * result)
{
   float4 mat = t_m_a;
   result[0] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
   mat = t_m_b;
   result[1] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
   mat = t_m_c;
   result[2] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
}
/* *************************************************************** */
__inline__ __device__
float warpAllReduceSum(float val)
{
   for (int mask = 16; mask > 0; mask /= 2)
      val += __shfl_xor(val, mask);
   return val;
}
/* *************************************************************** */
__inline__ __device__
float warpReduceSum(float val)
{
   for (int offset = 16; offset > 0; offset /= 2)
      val += __shfl_down(val, offset);
   return val;
}
/* *************************************************************** */
__inline__ __device__
float blockReduce2DSum(float val, int tid)
{
   static __shared__ float shared[2];
   int laneId = tid % 8;
   int warpId = tid / 8;

   val = warpReduceSum(val);     // Each warp performs partial reduction

   if (laneId == 0)
      shared[warpId] = val;
   //if (blockIdx.x == 8 && blockIdx.y == 0 && blockIdx.z == 0) printf("idx: %d | lane: %d \n", tid, lane);
   __syncthreads();

   return shared[0] + shared[1];
}
/* *************************************************************** */
__inline__ __device__
float blockReduceSum(float val, int tid)
{
   static __shared__ float shared[2];
   int laneId = tid % 32;
   int warpId = tid / 32;

   val = warpReduceSum(val);     // Each warp performs partial reduction

   if (laneId == 0)
      shared[warpId] = val;
   //if (blockIdx.x == 8 && blockIdx.y == 0 && blockIdx.z == 0) printf("idx: %d | lane: %d \n", tid, lane);
   __syncthreads();

   return shared[0] + shared[1];
}
/* *************************************************************** */
//recently switched to this kernel as it can accomodate greater capture range
__global__ void blockMatchingKernel2D(float *warpedPosition,
                                      float *referencePosition,
                                      int *mask,
                                      float* targetMatrix_xyz,
                                      unsigned int *definedBlock,
                                      uint3 c_ImageSize,
                                      const int blocksRange,
                                      const unsigned int stepSize)
{
   extern __shared__ float sResultValues[];

   const unsigned int numBlocks = blocksRange * 2 + 1;

   const unsigned int idy = threadIdx.x / 4;
   const unsigned int idx = threadIdx.x - 4 * idy;

   const unsigned int blockIndex = blockIdx.x + gridDim.x * blockIdx.y;

   const unsigned int xBaseImage = blockIdx.x * 4;
   const unsigned int yBaseImage = blockIdx.y * 4;

   const unsigned int tid = threadIdx.x;     //0-blockSize

   const unsigned int xImage = xBaseImage + idx;
   const unsigned int yImage = yBaseImage + idy;

   const unsigned long imgIdx = xImage + yImage * (c_ImageSize.x);
   const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y;

   const int currentBlockIndex = tex1Dfetch(totalBlock_texture, blockIndex);

   if (currentBlockIndex > -1) {

      float bestDisplacement[3] = { nanf("sNaN"), 0.0f, 0.0f };
      float bestCC = blocksRange > 1 ? 0.9f : 0.0f;

      //populate shared memory with resultImageArray's values
      for (int m = -1 * blocksRange; m <= blocksRange; m += 1) {
         for (int l = -1 * blocksRange; l <= blocksRange; l += 1) {
            const int x = l * 4 + idx;
            const int y = m * 4 + idy;

            const unsigned int sIdx = (y + blocksRange * 4) * numBlocks * 4 + (x + blocksRange * 4);

            const int xImageIn = xBaseImage + x;
            const int yImageIn = yBaseImage + y;

            const int indexXYZIn = xImageIn + yImageIn * (c_ImageSize.x);

            const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y);
            //sResultValues[sIdx] = (valid /*&& mask[indexXYZIn]>-1*/) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs
            sResultValues[sIdx] = (valid && mask[indexXYZIn] > -1) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs

         }
      }

      //for most cases we need this out of th loop
      //value if the block is 4x4x4 NaN otherwise
      float rTargetValue = (targetInBounds && mask[imgIdx] > -1) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");
      const bool finiteReferenceIntensity = isfinite(rTargetValue);
      rTargetValue = finiteReferenceIntensity ? rTargetValue : 0.f;

      const unsigned int targetBlockSize = __syncthreads_count(finiteReferenceIntensity);

      if (targetBlockSize > 8) {
         //the target values must remain constant throughout the block matching process
         const float referenceMean = __fdividef(blockReduce2DSum(rTargetValue, tid), targetBlockSize);
         const float referenceTemp = finiteReferenceIntensity ? rTargetValue - referenceMean : 0.f;
         const float referenceVar = blockReduce2DSum(referenceTemp * referenceTemp, tid);

         // iteration over the result blocks (block matching part)
         for (unsigned int m = 1; m < blocksRange * 8 /*2*4*/; m += stepSize) {
            for (unsigned int l = 1; l < blocksRange * 8 /*2*4*/; l += stepSize) {

               const unsigned int sIdxIn = (idy + m) * numBlocks * 4 + idx + l;

               const float rWarpedValue = sResultValues[sIdxIn];
               const bool overlap = isfinite(rWarpedValue) && finiteReferenceIntensity;
               const unsigned int blockSize = __syncthreads_count(overlap);

               if (blockSize > 8) {

                  //the target values must remain intact at each loop, so please do not touch this!
                  float newReferenceTemp = referenceTemp;
                  float newReferenceVar = referenceVar;
                  if (blockSize != targetBlockSize) {

                     const float newReferenceValue = overlap ? rTargetValue : 0.0f;
                     const float newReferenceMean = __fdividef(blockReduce2DSum(newReferenceValue, tid), blockSize);
                     newReferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
                     newReferenceVar = blockReduce2DSum(newReferenceTemp * newReferenceTemp, tid);
                  }

                  const float rChecked = overlap ? rWarpedValue : 0.0f;
                  const float warpedMean = __fdividef(blockReduce2DSum(rChecked, tid), blockSize);
                  const float warpedTemp = overlap ? rChecked - warpedMean : 0.0f;
                  float warpedVar = blockReduce2DSum(warpedTemp * warpedTemp, tid);

                  const float sumReferenceResult = blockReduce2DSum((newReferenceTemp)* (warpedTemp), tid);

                  //To be consistent with the variables name
                  newReferenceVar = newReferenceVar / blockSize;
                  warpedVar = warpedVar / blockSize;

                  const float localCC = sumReferenceResult > 0.0 ? fabs((sumReferenceResult / blockSize) * rsqrtf(newReferenceVar * warpedVar)) : 0;

                  if (tid == 0 && localCC > bestCC) {
                     bestCC = localCC;
                     bestDisplacement[0] = l - blocksRange * 4.0f;
                     bestDisplacement[1] = m - blocksRange * 4.0f;
                     bestDisplacement[2] = 0.0;
                  }
               }
            }
         }
      }
      if (tid == 0 /*&& isfinite(bestDisplacement[0])*/) {
         const unsigned int posIdx = 2 * currentBlockIndex;

         const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float)0 };

         bestDisplacement[0] += referencePosition_temp[0];
         bestDisplacement[1] += referencePosition_temp[1];
         bestDisplacement[2] += 0;

         reg2D_mat44_mul_cuda<float>(targetMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
         reg2D_mat44_mul_cuda<float>(targetMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);
         if (isfinite(bestDisplacement[0])) {
            atomicAdd(definedBlock, 1);
         }
      }
   }
}
/* *************************************************************** */
//recently switched to this kernel as it can accomodate greater capture range
__global__ void blockMatchingKernel3D(float *warpedPosition,
                                      float *referencePosition,
                                      int *mask,
                                      float* targetMatrix_xyz,
                                      unsigned int *definedBlock,
                                      uint3 c_ImageSize,
                                      const int blocksRange,
                                      const unsigned int stepSize)
{
   extern __shared__ float sResultValues[];

   const unsigned int numBlocks = blocksRange * 2 + 1;

   const unsigned int idz = threadIdx.x / 16;
   const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
   const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

   const unsigned int blockIndex = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

   const unsigned int xBaseImage = blockIdx.x * 4;
   const unsigned int yBaseImage = blockIdx.y * 4;
   const unsigned int zBaseImage = blockIdx.z * 4;

   const unsigned int tid = threadIdx.x;     //0-blockSize

   const unsigned int xImage = xBaseImage + idx;
   const unsigned int yImage = yBaseImage + idy;
   const unsigned int zImage = zBaseImage + idz;

   const unsigned long imgIdx = xImage + yImage * (c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
   const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

   const int currentBlockIndex = tex1Dfetch(totalBlock_texture, blockIndex);

   if (currentBlockIndex > -1) {

      float bestDisplacement[3] = { nanf("sNaN"), 0.0f, 0.0f };
      float bestCC = blocksRange > 1 ? 0.9f : 0.0f;

      //populate shared memory with resultImageArray's values
      for (int n = -1 * blocksRange; n <= blocksRange; n += 1) {
         for (int m = -1 * blocksRange; m <= blocksRange; m += 1) {
            for (int l = -1 * blocksRange; l <= blocksRange; l += 1) {
               const int x = l * 4 + idx;
               const int y = m * 4 + idy;
               const int z = n * 4 + idz;

               const unsigned int sIdx = (z + blocksRange * 4) * numBlocks * 4 * numBlocks * 4 + (y + blocksRange * 4) * numBlocks * 4 + (x + blocksRange * 4);

               const int xImageIn = xBaseImage + x;
               const int yImageIn = yBaseImage + y;
               const int zImageIn = zBaseImage + z;

               const int indexXYZIn = xImageIn + yImageIn * (c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

               const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
               //sResultValues[sIdx] = (valid /*&& mask[indexXYZIn]>-1*/) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs
               sResultValues[sIdx] = (valid && mask[indexXYZIn] > -1) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs

            }
         }
      }

      //for most cases we need this out of th loop
      //value if the block is 4x4x4 NaN otherwise
      float rTargetValue = (targetInBounds && mask[imgIdx] > -1) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");
      const bool finiteTargetIntensity = isfinite(rTargetValue);
      rTargetValue = finiteTargetIntensity ? rTargetValue : 0.f;

      const unsigned int targetBlockSize = __syncthreads_count(finiteTargetIntensity);

      if (targetBlockSize > 32) {
         //the target values must remain constant throughout the block matching process
         const float targetMean = __fdividef(blockReduceSum(rTargetValue, tid), targetBlockSize);
         const float targetTemp = finiteTargetIntensity ? rTargetValue - targetMean : 0.f;
         const float targetVar = blockReduceSum(targetTemp * targetTemp, tid);

         // iteration over the result blocks (block matching part)
         for (unsigned int n = 1; n < blocksRange * 8 /*2*4*/; n += stepSize) {
            for (unsigned int m = 1; m < blocksRange * 8 /*2*4*/; m += stepSize) {
               for (unsigned int l = 1; l < blocksRange * 8 /*2*4*/; l += stepSize) {

                  const unsigned int sIdxIn = (idz + n) * numBlocks * 4 * numBlocks * 4 + (idy + m) * numBlocks * 4 + idx + l;
                  const float rResultValue = sResultValues[sIdxIn];
                  const bool overlap = isfinite(rResultValue) && finiteTargetIntensity;
                  const unsigned int blockSize = __syncthreads_count(overlap);

                  if (blockSize > 32) {

                     //the target values must remain intact at each loop, so please do not touch this!
                     float newTargetTemp = targetTemp;
                     float newTargetVar = targetVar;
                     if (blockSize != targetBlockSize) {

                        const float newTargetValue = overlap ? rTargetValue : 0.0f;
                        const float newTargetMean = __fdividef(blockReduceSum(newTargetValue, tid), blockSize);
                        newTargetTemp = overlap ? newTargetValue - newTargetMean : 0.0f;
                        newTargetVar = blockReduceSum(newTargetTemp * newTargetTemp, tid);
                     }

                     const float rChecked = overlap ? rResultValue : 0.0f;
                     const float resultMean = __fdividef(blockReduceSum(rChecked, tid), blockSize);
                     const float resultTemp = overlap ? rChecked - resultMean : 0.0f;
                     float resultVar = blockReduceSum(resultTemp * resultTemp, tid);

                     const float sumTargetResult = blockReduceSum((newTargetTemp)* (resultTemp), tid);

                     //To be consistent with the variables name
                     newTargetVar = newTargetVar / blockSize;
                     resultVar = resultVar / blockSize;

                     const float localCC = sumTargetResult > 0.0 ? fabs((sumTargetResult / blockSize) * rsqrtf(newTargetVar * resultVar)) : 0;

                     if (tid == 0 && localCC > bestCC) {
                        bestCC = localCC;
                        bestDisplacement[0] = l - blocksRange * 4.0f;
                        bestDisplacement[1] = m - blocksRange * 4.0f;
                        bestDisplacement[2] = n - blocksRange * 4.0f;
                     }
                  }
               }
            }
         }
      }
      if (tid == 0 /*&& isfinite(bestDisplacement[0])*/) {
         const unsigned int posIdx = 3 * currentBlockIndex;

         const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float)zBaseImage };

         bestDisplacement[0] += referencePosition_temp[0];
         bestDisplacement[1] += referencePosition_temp[1];
         bestDisplacement[2] += referencePosition_temp[2];

         reg_mat44_mul_cuda<float>(targetMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
         reg_mat44_mul_cuda<float>(targetMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);
         if (isfinite(bestDisplacement[0])){
            atomicAdd(definedBlock, 1);
         }
      }
   }
}
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/* *************************************************************** */
void block_matching_method_gpu(nifti_image *targetImage,
										 _reg_blockMatchingParam *params,
										 float **targetImageArray_d,
										 float **resultImageArray_d,
										 float **referencePosition_d,
										 float **warpedPosition_d,
										 int **totalBlock_d,
										 int **mask_d,
										 float** referenceMat_d)
{
	// Copy some required parameters over to the device
	uint3 imageSize = make_uint3(targetImage->nx, targetImage->ny, targetImage->nz); // Image size

	// Texture binding
	const unsigned int numBlocks = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, totalBlock_texture, *totalBlock_d, numBlocks * sizeof(int)));

	unsigned int *definedBlock_d;
	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	NR_CUDA_SAFE_CALL(cudaMalloc((void** )(&definedBlock_d), sizeof(unsigned int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(definedBlock_d, definedBlock_h, sizeof(unsigned int), cudaMemcpyHostToDevice));

	const int blockRange = params->voxelCaptureRange % 4 ? params->voxelCaptureRange / 4 + 1 : params->voxelCaptureRange / 4;
	dim3 BlockDims1D(64,1,1);
	dim3 BlocksGrid3D(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);

	if (targetImage->nz == 1){

		BlockDims1D.x=16;

		const unsigned int sMem = (blockRange * 2 + 1) * (blockRange * 2 + 1) * 16 * sizeof(float);
		blockMatchingKernel2D << <BlocksGrid3D, BlockDims1D, sMem >> >(*warpedPosition_d,
																							*referencePosition_d,
																							*mask_d,
																							*referenceMat_d,
																							definedBlock_d,
																							imageSize,
																							blockRange,
																							params->stepSize);
	}
	else {

		const unsigned int sMem = (blockRange * 2 + 1) * (blockRange * 2 + 1) * (blockRange * 2 + 1) * 64 * sizeof(float);
		blockMatchingKernel3D<< <BlocksGrid3D, BlockDims1D, sMem >> >(*warpedPosition_d,
																						  *referencePosition_d,
																						  *mask_d,
																						  *referenceMat_d,
																						  definedBlock_d,
																						  imageSize,
																						  blockRange,
																						  params->stepSize);
	}
#ifndef NDEBUG
	NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D)
#else
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif

	NR_CUDA_SAFE_CALL(cudaMemcpy((void * )definedBlock_h, (void * )definedBlock_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	params->definedActiveBlockNumber = *definedBlock_h;
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(totalBlock_texture));

	free(definedBlock_h);
	cudaFree(definedBlock_d);

}
/* *************************************************************** */
#endif //_REG_BLOCKMATCHING_GPU_CU
