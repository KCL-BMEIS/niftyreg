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

//#define USE_TEST_KERNEL
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
__device__ __constant__ int3 c_BlockDim;
__device__ __constant__ uint3 c_ImageSize;

// Transformation matrix from nifti header
__device__          __constant__ float4 t_m_a;
__device__          __constant__ float4 t_m_b;
__device__          __constant__ float4 t_m_c;

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define OVERLAP_SIZE 3
#define STEP_SIZE 1


texture<float, 1, cudaReadModeElementType> referenceImageArray_texture;
texture<float, 1, cudaReadModeElementType> warpedImageArray_texture;
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
float blockReduce2DSum(float val, int tid)
{
   static __shared__ float shared[16];
   shared[tid] = val;
   __syncthreads();

	for (unsigned int i = 8; i > 0; i >>= 1){
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
		__syncthreads();
	}
	return shared[0];
}
/* *************************************************************** */
__inline__ __device__
float blockReduceSum(float val, int tid)
{
   static __shared__ float shared[64];
   shared[tid] = val;
   __syncthreads();

	for (unsigned int i = 32; i > 0; i >>= 1){
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
		__syncthreads();
	}
	return shared[0];
}
/* *************************************************************** */
__global__ void blockMatchingKernel2D(float *warpedPosition,
                                      float *referencePosition,
                                      int *mask,
                                      float* referenceMatrix_xyz,
                                      unsigned int *definedBlock)
{
	extern __shared__ float sWarpedValues[];
	// Compute the current block index
    const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

	const int currentBlockIndex = tex1Dfetch(totalBlock_texture, bid);
	if (currentBlockIndex > -1) {

		const unsigned int idy = threadIdx.x;
		const unsigned int idx = threadIdx.y;
		const unsigned int tid = idy * 4 + idx;

		const unsigned int xImage = blockIdx.x * 4 + idx;
		const unsigned int yImage = blockIdx.y * 4 + idy;

		//populate shared memory with resultImageArray's values
		for (int y=-1; y<2; ++y) {
			const int yImageIn = yImage + y * 4;
			for (int x=-1; x<2; ++x) {
				const int xImageIn = xImage + x * 4;

				const int sharedIndex = ((y+1)*4+idy)*12+(x+1)*4+idx;

				const int indexXYIn = yImageIn * c_ImageSize.x + xImageIn;

				const bool valid =
						(xImageIn > -1 && xImageIn < (int)c_ImageSize.x) &&
						(yImageIn > -1 && yImageIn < (int)c_ImageSize.y);
				sWarpedValues[sharedIndex] = (valid && mask[indexXYIn] > -1) ?
							tex1Dfetch(warpedImageArray_texture, indexXYIn) : nanf("sNaN");
			}
		}

		//for most cases we need this out of th loop
		//value if the block is 4x4 NaN otherwise
		const unsigned long voxIndex = yImage * c_ImageSize.x + xImage;
		const bool referenceInBounds =
				xImage < c_ImageSize.x &&
				yImage < c_ImageSize.y;
		float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
					tex1Dfetch(referenceImageArray_texture, voxIndex) : nanf("sNaN");
		const bool finiteReference = isfinite(rReferenceValue);
		rReferenceValue = finiteReference ? rReferenceValue : 0.f;
		const unsigned int referenceSize = __syncthreads_count(finiteReference);

        float bestDisplacement[2] = {nanf("sNaN"), 0.0f};
        float bestCC = 0.0;

		if (referenceSize > 8) {
			//the target values must remain constant throughout the block matching process
			const float referenceMean = __fdividef(blockReduce2DSum(rReferenceValue, tid), referenceSize);
			const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.f;
			const float referenceVar = blockReduce2DSum(referenceTemp * referenceTemp, tid);
			// iteration over the result blocks (block matching part)
			for (unsigned int y=1; y<8; ++y) {
				for (unsigned int x=1; x<8; ++x) {

					const unsigned int sharedIndex = ( y + idy ) * 12 + x + idx;
					const float rWarpedValue = sWarpedValues[sharedIndex];
					const bool overlap = isfinite(rWarpedValue) && finiteReference;
					const unsigned int currentWarpedSize = __syncthreads_count(overlap);

                    if (currentWarpedSize > 8) {
                        //the reference values must remain intact at each loop, so please do not touch this!
						float newreferenceTemp = referenceTemp;
						float newreferenceVar = referenceVar;
						if (currentWarpedSize != referenceSize){
							const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
							const float newReferenceMean = __fdividef(blockReduce2DSum(newReferenceValue, tid), currentWarpedSize);
							newreferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
							newreferenceVar = blockReduce2DSum(newreferenceTemp * newreferenceTemp, tid);
						}

						const float rChecked = overlap ? rWarpedValue : 0.0f;
						const float warpedMean = __fdividef(blockReduce2DSum(rChecked, tid), currentWarpedSize);
						const float warpedTemp = overlap ? rChecked - warpedMean : 0.0f;
						const float warpedVar = blockReduce2DSum(warpedTemp * warpedTemp, tid);

						const float sumTargetResult = blockReduce2DSum((newreferenceTemp)* (warpedTemp), tid);
                        const float localCC = (newreferenceVar * warpedVar) > 0.0 ? fabs((sumTargetResult) / sqrt(newreferenceVar * warpedVar)) : 0.0;

                        if (tid == 0 && localCC > bestCC) {
                            bestCC = localCC + 1.0e-7f;
                            bestDisplacement[0] = x - 4.f;
                            bestDisplacement[1] = y - 4.f;
                        }
					}
				}
			}
		}

        if (tid==0){
			const unsigned int posIdx = 2 * currentBlockIndex;
			const float referencePosition_temp[2] = {(float)xImage, (float)yImage};

			bestDisplacement[0] += referencePosition_temp[0];
			bestDisplacement[1] += referencePosition_temp[1];

			reg2D_mat44_mul_cuda<float>(referenceMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
            reg2D_mat44_mul_cuda<float>(referenceMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);

			if (isfinite(bestDisplacement[0])) {
				atomicAdd(definedBlock, 1);
			}
		}
	}
}
/* *************************************************************** */
#ifdef USE_TEST_KERNEL
__inline__ __device__
float2 REDUCE_TEST(float* sData,
                   float data,
                   unsigned int tid)
{
	sData[tid] = data;
	__syncthreads();

	bool seconHalf = tid > 63 ? true : false;
	for (unsigned int i = 32; i > 0; i >>= 1){
		if (tid < i) sData[tid] += sData[tid + i];
		if (seconHalf && tid < 64 + i) sData[tid] += sData[tid + i];
		__syncthreads();
	}

	const float2 temp = make_float2(sData[0], sData[64]);
	__syncthreads();
	return temp;
}
/* *************************************************************** */
__global__ void blockMatchingKernel3D(float *warpedPosition,
                                      float *referencePosition,
                                      int *mask,
                                      float* referenceMatrix_xyz,
                                      unsigned int *definedBlock)
{
   extern __shared__ float sWarpedValues[];
   float *sData = &sWarpedValues[12*12*16];

   // Compute the current block index
   const unsigned int bid0 = (2*blockIdx.z * gridDim.y + blockIdx.y) *
         gridDim.x + blockIdx.x;
   const unsigned int bid1 = bid0 + gridDim.x * gridDim.y;
   int currentBlockIndex[2] = {tex1Dfetch(totalBlock_texture, bid0),
                               tex1Dfetch(totalBlock_texture, bid1)};
   currentBlockIndex[1] = (2*blockIdx.z+1)<c_BlockDim.z ? currentBlockIndex[1] : -1;
   if (currentBlockIndex[0] > -1 || currentBlockIndex[1] > -1) {
      const unsigned int idx = threadIdx.x;
      const unsigned int idy = threadIdx.y;
      const unsigned int idz = threadIdx.z;
      const unsigned int tid = (idz*4+idy)*4+idx;
      const unsigned int xImage = blockIdx.x * 4 + idx;
      const unsigned int yImage = blockIdx.y * 4 + idy;
      const unsigned int zImage = blockIdx.z * 8 + idz;

      //populate shared memory with resultImageArray's values
      for (int z=-1 ; z<2; z+=2) {
         const int zImageIn = zImage + z * 4;
         for (int y=-1; y<2; ++y) {
            const int yImageIn = yImage + y * 4;
            for (int x=-1; x<2; ++x) {
               const int xImageIn = xImage + x * 4;

               const int sharedIndex = (((z+1)*4+idz)*12+(y+1)*4+idy)*12+(x+1)*4+idx;

               const unsigned int indexXYZIn = xImageIn + c_ImageSize.x *
                     (yImageIn + zImageIn * c_ImageSize.y);

               const bool valid =
                     (xImageIn > -1 && xImageIn < (int)c_ImageSize.x) &&
                     (yImageIn > -1 && yImageIn < (int)c_ImageSize.y) &&
                     (zImageIn > -1 && zImageIn < (int)c_ImageSize.z);
               sWarpedValues[sharedIndex] = (valid && mask[indexXYZIn] > -1) ?
                        tex1Dfetch(warpedImageArray_texture, indexXYZIn) : nanf("sNaN");
            }
         }
      }

      const unsigned int voxIndex = ( zImage * c_ImageSize.y + yImage ) *
            c_ImageSize.x + xImage;
      const bool referenceInBounds =
            xImage < c_ImageSize.x &&
            yImage < c_ImageSize.y &&
            zImage < c_ImageSize.z;
      float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
               tex1Dfetch(referenceImageArray_texture, voxIndex) : nanf("sNaN");
      const bool finiteReference = isfinite(rReferenceValue);
      rReferenceValue = finiteReference ? rReferenceValue : 0.f;
      float2 tempVal = REDUCE_TEST(sData, finiteReference ? 1.0f : 0.0f, tid);
      const uint2 referenceSize = make_uint2((uint)tempVal.x, (uint)tempVal.y);

      float2 bestValue = make_float2(0.f, 0.f);
      float bestDisp[2][3];
      bestDisp[0][0] = bestDisp[1][0] = nanf("sNaN");
      if (referenceSize.x > 32 || referenceSize.y > 32) {
         float2 referenceMean=REDUCE_TEST(sData, rReferenceValue, tid);
         referenceMean.x /= (float)referenceSize.x;
         referenceMean.y /= (float)referenceSize.y;
         float referenceTemp;
         if(tid>63)
            referenceTemp = finiteReference ? rReferenceValue - referenceMean.y : 0.f;
         else referenceTemp = finiteReference ? rReferenceValue - referenceMean.x : 0.f;
         float2 referenceVar = REDUCE_TEST(sData, referenceTemp*referenceTemp, tid);

         // iteration over the result blocks (block matching part)
         for (unsigned int z=1; z<8; ++z) {
            for (unsigned int y=1; y<8; ++y) {
               for (unsigned int x=1; x<8; ++x) {

                  const unsigned int sharedIndex = ( (z+idz) * 12 + y + idy ) * 12 + x + idx;
                  const float rWarpedValue = sWarpedValues[sharedIndex];
                  const bool overlap = isfinite(rWarpedValue) && finiteReference;
                  tempVal = REDUCE_TEST(sData, overlap ? 1.0f : 0.0f, tid);
                  const uint2 currentWarpedSize = make_uint2((uint)tempVal.x, (uint)tempVal.y);

                  if (currentWarpedSize.x > 32 || currentWarpedSize.y > 32) {

                     float newreferenceTemp = referenceTemp;
                     float2 newreferenceVar = referenceVar;
                     if (currentWarpedSize.x!=referenceSize.x || currentWarpedSize.y!=referenceSize.y){
                        const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
                        float2 newReferenceMean = REDUCE_TEST(sData, newReferenceValue, tid);
                        newReferenceMean.x /= (float)currentWarpedSize.x;
                        newReferenceMean.y /= (float)currentWarpedSize.y;
                        if(tid>63)
                           referenceTemp = overlap ? newReferenceValue - newReferenceMean.y : 0.f;
                        else referenceTemp = overlap ? newReferenceValue - newReferenceMean.x : 0.f;
                        newreferenceVar = REDUCE_TEST(sData, newreferenceTemp * newreferenceTemp, tid);
                     }
                     const float rChecked = overlap ? rWarpedValue : 0.0f;
                     float2 warpedMean = REDUCE_TEST(sData, rChecked, tid);
                     warpedMean.x /= (float)currentWarpedSize.x;
                     warpedMean.y /= (float)currentWarpedSize.y;
                     float warpedTemp;
                     if(tid>63)
                        warpedTemp = overlap ? rChecked - warpedMean.y : 0.f;
                     else warpedTemp = overlap ? rChecked - warpedMean.x : 0.f;
                     const float2 warpedVar = REDUCE_TEST(sData, warpedTemp*warpedTemp, tid);
                     const float2 sumTargetResult = REDUCE_TEST(sData, newreferenceTemp*warpedTemp, tid);

                     if (tid==0 && currentWarpedSize.x > 32 ){
                        const float localCC = fabs(sumTargetResult.x *
                                                   rsqrtf(newreferenceVar.x * warpedVar.x));
                        if(localCC > bestValue.x) {
                           bestValue.x = localCC;
                           bestDisp[0][0] = x - 4.f;
                           bestDisp[0][1] = y - 4.f;
                           bestDisp[0][2] = z - 4.f;
                        }
                     }
                     if (tid==64 && currentWarpedSize.y > 32 ){
                        const float localCC = fabs(sumTargetResult.y *
                                                   rsqrtf(newreferenceVar.y * warpedVar.y));
                        if(localCC > bestValue.y) {
                           bestValue.y = localCC;
                           bestDisp[1][0] = x - 4.f;
                           bestDisp[1][1] = y - 4.f;
                           bestDisp[1][2] = z - 4.f;
                        }
                     }
                     __syncthreads();
                  }
               }
            }
         }
      }

      if(tid==0 && currentBlockIndex[0]>-1){
         const unsigned int posIdx = 3 * currentBlockIndex[0];
         warpedPosition[posIdx] = NAN;
         if (isfinite(bestDisp[0][0])){
            const float referencePosition_temp[3] = { (float)xImage,
                                                      (float)yImage,
                                                      (float)zImage};
            bestDisp[0][0] += referencePosition_temp[0];
            bestDisp[0][1] += referencePosition_temp[1];
            bestDisp[0][2] += referencePosition_temp[2];
            reg_mat44_mul_cuda<float>(referenceMatrix_xyz,
                                      referencePosition_temp,
                                      &referencePosition[posIdx]);
            reg_mat44_mul_cuda<float>(referenceMatrix_xyz,
                                      bestDisp[0],
                  &warpedPosition[posIdx]);
            atomicAdd(definedBlock, 1);
         }
      }
      if(tid==64 && currentBlockIndex[1]>-1){
         const unsigned int posIdx = 3 * currentBlockIndex[1];
         warpedPosition[posIdx] = NAN;
         if (isfinite(bestDisp[1][0])){
            const float referencePosition_temp[3] = {(float)xImage,
                                                     (float)yImage,
                                                     (float)zImage};
            bestDisp[1][0] += referencePosition_temp[0];
            bestDisp[1][1] += referencePosition_temp[1];
            bestDisp[1][2] += referencePosition_temp[2];
            reg_mat44_mul_cuda<float>(referenceMatrix_xyz,
                                      referencePosition_temp,
                                      &referencePosition[posIdx]);
            reg_mat44_mul_cuda<float>(referenceMatrix_xyz,
                                      bestDisp[1],
                  &warpedPosition[posIdx]);
            atomicAdd(definedBlock, 1);
         }
      }
   }
}
#else

/* *************************************************************** */
__global__ void blockMatchingKernel3D(float *warpedPosition,
                                      float *referencePosition,
                                      int *mask,
                                      float* referenceMatrix_xyz,
                                      unsigned int *definedBlock)
{
	extern __shared__ float sWarpedValues[];
	// Compute the current block index
	const unsigned int bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x ;

	const int currentBlockIndex = tex1Dfetch(totalBlock_texture, bid);
	if (currentBlockIndex > -1) {
		const unsigned int idx = threadIdx.x;
		const unsigned int idy = threadIdx.y;
		const unsigned int idz = threadIdx.z;
		const unsigned int tid = (idz*4+idy)*4+idx;
		const unsigned int xImage = blockIdx.x * 4 + idx;
		const unsigned int yImage = blockIdx.y * 4 + idy;
		const unsigned int zImage = blockIdx.z * 4 + idz;

		//populate shared memory with resultImageArray's values
		for (int z=-1 ; z<2; ++z) {
			const int zImageIn = zImage + z * 4;
			for (int y=-1; y<2; ++y) {
				const int yImageIn = yImage + y * 4;
				for (int x=-1; x<2; ++x) {
					const int xImageIn = xImage + x * 4;

					const int sharedIndex = (((z+1)*4+idz)*12+(y+1)*4+idy)*12+(x+1)*4+idx;

					const unsigned int indexXYZIn = xImageIn + c_ImageSize.x *
							(yImageIn + zImageIn * c_ImageSize.y);

					const bool valid =
							(xImageIn > -1 && xImageIn < (int)c_ImageSize.x) &&
							(yImageIn > -1 && yImageIn < (int)c_ImageSize.y) &&
							(zImageIn > -1 && zImageIn < (int)c_ImageSize.z);
					sWarpedValues[sharedIndex] = (valid && mask[indexXYZIn] > -1) ?
								tex1Dfetch(warpedImageArray_texture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs
				}
			}
		}

		//for most cases we need this out of th loop
		//value if the block is 4x4x4 NaN otherwise
		const unsigned int voxIndex = ( zImage * c_ImageSize.y + yImage ) *
				c_ImageSize.x + xImage;
		const bool referenceInBounds =
				xImage < c_ImageSize.x &&
				yImage < c_ImageSize.y &&
				zImage < c_ImageSize.z;
		float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
					tex1Dfetch(referenceImageArray_texture, voxIndex) : nanf("sNaN");
		const bool finiteReference = isfinite(rReferenceValue);
		rReferenceValue = finiteReference ? rReferenceValue : 0.f;
		const unsigned int referenceSize = __syncthreads_count(finiteReference);

        float bestDisplacement[3] = {nanf("sNaN"), 0.0f, 0.0f };
        float bestCC = 0.0f;

		if (referenceSize > 32) {
			//the target values must remain constant throughout the block matching process
			const float referenceMean = __fdividef(blockReduceSum(rReferenceValue, tid), referenceSize);
			const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.f;
			const float referenceVar = blockReduceSum(referenceTemp * referenceTemp, tid);

			// iteration over the result blocks (block matching part)
			for (unsigned int z=1; z<8; ++z) {
				for (unsigned int y=1; y<8; ++y) {
					for (unsigned int x=1; x<8; ++x) {

						const unsigned int sharedIndex = ( (z+idz) * 12 + y + idy ) * 12 + x + idx;
						const float rWarpedValue = sWarpedValues[sharedIndex];
						const bool overlap = isfinite(rWarpedValue) && finiteReference;
						const unsigned int currentWarpedSize = __syncthreads_count(overlap);

						if (currentWarpedSize > 32) {

							//the target values must remain intact at each loop, so please do not touch this!
							float newreferenceTemp = referenceTemp;
							float newreferenceVar = referenceVar;
							if (currentWarpedSize != referenceSize){
								const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
								const float newReferenceMean = __fdividef(blockReduceSum(newReferenceValue, tid), currentWarpedSize);
								newreferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
								newreferenceVar = blockReduceSum(newreferenceTemp * newreferenceTemp, tid);
							}

							const float rChecked = overlap ? rWarpedValue : 0.0f;
							const float warpedMean = __fdividef(blockReduceSum(rChecked, tid), currentWarpedSize);
							const float warpedTemp = overlap ? rChecked - warpedMean : 0.0f;
							const float warpedVar = blockReduceSum(warpedTemp * warpedTemp, tid);

							const float sumTargetResult = blockReduceSum((newreferenceTemp)* (warpedTemp), tid);
                            const float localCC = (newreferenceVar * warpedVar) > 0.0 ? fabs((sumTargetResult) / sqrt(newreferenceVar * warpedVar)) : 0.0;

                            if (tid == 0 && localCC > bestCC) {
                                bestCC = localCC + 1.0e-7f;
                                bestDisplacement[0] = x - 4.f;
                                bestDisplacement[1] = y - 4.f;
                                bestDisplacement[2] = z - 4.f;
                            }
						}
					}
				}
			}
		}

		if (tid==0) {
			const unsigned int posIdx = 3 * currentBlockIndex;
			const float referencePosition_temp[3] = { (float)xImage, (float)yImage, (float)zImage };

			bestDisplacement[0] += referencePosition_temp[0];
			bestDisplacement[1] += referencePosition_temp[1];
			bestDisplacement[2] += referencePosition_temp[2];

			reg_mat44_mul_cuda<float>(referenceMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
			reg_mat44_mul_cuda<float>(referenceMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);
			if (isfinite(bestDisplacement[0])) {
				atomicAdd(definedBlock, 1);
			}
		}
	}
}
#endif
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
    uint3 imageSize = make_uint3(targetImage->nx,
                                 targetImage->ny,
                                 targetImage->nz);
	uint3 blockSize = make_uint3(params->blockNumber[0],
			params->blockNumber[1],
			params->blockNumber[2]);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize,&imageSize,sizeof(uint3)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_BlockDim,&blockSize,sizeof(uint3)));

	// Texture binding
	const unsigned int numBlocks = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, referenceImageArray_texture, *targetImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, warpedImageArray_texture, *resultImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, totalBlock_texture, *totalBlock_d, numBlocks * sizeof(int)));

	unsigned int *definedBlock_d;
	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	NR_CUDA_SAFE_CALL(cudaMalloc((void** )(&definedBlock_d), sizeof(unsigned int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(definedBlock_d, definedBlock_h, sizeof(unsigned int), cudaMemcpyHostToDevice));


	if (params->stepSize!=1 || params->voxelCaptureRange!=3){
        reg_print_msg_error("The block Mathching CUDA kernel supports only a stepsize of 1");
		reg_exit();
	}

#ifdef USE_TEST_KERNEL
	dim3 BlockDims1D(4,4,8);
	dim3 BlocksGrid3D(
				params->blockNumber[0],
			params->blockNumber[1],
			(unsigned int)reg_ceil((float)params->blockNumber[2]/2.f));
	unsigned int sMem = (128 + 4*3 * 4*3 * 4*4) * sizeof(float);
#else
    dim3 BlockDims1D(4,4,4);
    dim3 BlocksGrid3D(
                params->blockNumber[0],
            params->blockNumber[1],
            params->blockNumber[2]);
    unsigned int sMem = (64 + 4*3 * 4*3 * 4*3) * sizeof(float); // (3*4)^3
#endif

	if (targetImage->nz == 1){
		BlockDims1D.z=1;
		BlocksGrid3D.z=1;
		sMem = (16 + 144) * sizeof(float); // // (3*4)^2
		blockMatchingKernel2D << <BlocksGrid3D, BlockDims1D, sMem >> >(*warpedPosition_d,
																							*referencePosition_d,
																							*mask_d,
																							*referenceMat_d,
																							definedBlock_d);
	}
	else {
		blockMatchingKernel3D <<<BlocksGrid3D, BlockDims1D, sMem>>>(*warpedPosition_d,
																						*referencePosition_d,
																						*mask_d,
																						*referenceMat_d,
																						definedBlock_d);
	}
#ifndef NDEBUG
    NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D);
        #else
    NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif

	NR_CUDA_SAFE_CALL(cudaMemcpy((void * )definedBlock_h, (void * )definedBlock_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	params->definedActiveBlockNumber = *definedBlock_h;
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(referenceImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(warpedImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(totalBlock_texture));

	free(definedBlock_h);
	cudaFree(definedBlock_d);

}
/* *************************************************************** */
#endif //_REG_BLOCKMATCHING_GPU_CU
