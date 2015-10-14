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

#ifdef CUDA7
	#include "cublas_v2.h"
	#include "cusolverDn.h"
	#include "nvToolsExt.h"
	#include "nvToolsExtCuda.h"
#endif
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

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

texture<float, 1, cudaReadModeElementType> targetImageArray_texture;
texture<float, 1, cudaReadModeElementType> resultImageArray_texture;
texture<int, 1, cudaReadModeElementType> totalBlock_texture;
/* *************************************************************** */
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
template<class DTYPE>
__device__ __inline__
void reg2D_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
    out[0] = (DTYPE)mat[0 * 4 + 0] * in[0] + (DTYPE)mat[0 * 4 + 1] * in[1] + (DTYPE)mat[0 * 4 + 2] * 0 + (DTYPE)mat[0 * 4 + 3];
    out[1] = (DTYPE)mat[1 * 4 + 0] * in[0] + (DTYPE)mat[1 * 4 + 1] * in[1] + (DTYPE)mat[1 * 4 + 2] * 0 + (DTYPE)mat[1 * 4 + 3];
    return;
}
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__
void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
    out[0] = (DTYPE)mat[0 * 4 + 0] * in[0] + (DTYPE)mat[0 * 4 + 1] * in[1] + (DTYPE)mat[0 * 4 + 2] * in[2] + (DTYPE)mat[0 * 4 + 3];
    out[1] = (DTYPE)mat[1 * 4 + 0] * in[0] + (DTYPE)mat[1 * 4 + 1] * in[1] + (DTYPE)mat[1 * 4 + 2] * in[2] + (DTYPE)mat[1 * 4 + 3];
    out[2] = (DTYPE)mat[2 * 4 + 0] * in[0] + (DTYPE)mat[2 * 4 + 1] * in[1] + (DTYPE)mat[2 * 4 + 2] * in[2] + (DTYPE)mat[2 * 4 + 3];
    return;
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

    float* start_warpedPosition = &warpedPosition[0];
    float* start_referencePosition = &referencePosition[0];

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
        const bool finiteTargetIntensity = isfinite(rTargetValue);
        rTargetValue = finiteTargetIntensity ? rTargetValue : 0.f;

        const unsigned int targetBlockSize = __syncthreads_count(finiteTargetIntensity);

        if (targetBlockSize > 8) {
            //the target values must remain constant throughout the block matching process
            const float targetMean = __fdividef(blockReduce2DSum(rTargetValue, tid), targetBlockSize);
            const float targetTemp = finiteTargetIntensity ? rTargetValue - targetMean : 0.f;
            const float targetVar = blockReduce2DSum(targetTemp * targetTemp, tid);

            // iteration over the result blocks (block matching part)
            for (unsigned int m = 1; m < blocksRange * 8 /*2*4*/; m += stepSize) {
                for (unsigned int l = 1; l < blocksRange * 8 /*2*4*/; l += stepSize) {

                    const unsigned int sIdxIn = (idy + m) * numBlocks * 4 + idx + l;
                    const float rResultValue = sResultValues[sIdxIn];
                    const bool overlap = isfinite(rResultValue) && finiteTargetIntensity;
                    const unsigned int blockSize = __syncthreads_count(overlap);

                    if (blockSize > 8) {

                        //the target values must remain intact at each loop, so please do not touch this!
                        float newTargetTemp = targetTemp;
                        float newTargetVar = targetVar;
                        if (blockSize != targetBlockSize) {

                            const float newTargetValue = overlap ? rTargetValue : 0.0f;
                            const float newTargetMean = __fdividef(blockReduce2DSum(newTargetValue, tid), blockSize);
                            newTargetTemp = overlap ? newTargetValue - newTargetMean : 0.0f;
                            newTargetVar = blockReduce2DSum(newTargetTemp * newTargetTemp, tid);
                        }

                        const float rChecked = overlap ? rResultValue : 0.0f;
                        const float resultMean = __fdividef(blockReduce2DSum(rChecked, tid), blockSize);
                        const float resultTemp = overlap ? rChecked - resultMean : 0.0f;
                        const float resultVar = blockReduce2DSum(resultTemp * resultTemp, tid);

                        const float sumTargetResult = blockReduce2DSum((newTargetTemp)* (resultTemp), tid);
                        const float localCC = fabs((sumTargetResult)* rsqrtf(newTargetVar * resultVar));

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

            referencePosition = start_referencePosition + posIdx;
            warpedPosition = start_warpedPosition + posIdx;

            const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float)0 };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] += 0;

            reg2D_mat44_mul_cuda<float>(targetMatrix_xyz, referencePosition_temp, referencePosition);
            reg2D_mat44_mul_cuda<float>(targetMatrix_xyz, bestDisplacement, warpedPosition);
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

    float* start_warpedPosition = &warpedPosition[0];
    float* start_referencePosition = &referencePosition[0];

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
                            const float resultVar = blockReduceSum(resultTemp * resultTemp, tid);

                            const float sumTargetResult = blockReduceSum((newTargetTemp)* (resultTemp), tid);
                            const float localCC = fabs((sumTargetResult)* rsqrtf(newTargetVar * resultVar));

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

            referencePosition = start_referencePosition + posIdx;
            warpedPosition = start_warpedPosition + posIdx;

            const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float)zBaseImage };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] += referencePosition_temp[2];

            reg_mat44_mul_cuda<float>(targetMatrix_xyz, referencePosition_temp, referencePosition);
            reg_mat44_mul_cuda<float>(targetMatrix_xyz, bestDisplacement, warpedPosition);
            if (isfinite(bestDisplacement[0])){
                atomicAdd(definedBlock, 1);
            }
        }
    }
}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void populateMatrixA(float* A, float *target, unsigned int numBlocks)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int c = tid * 3;
    //	const unsigned int n = 12;
    const unsigned int lda = numBlocks * 3;

    if (tid < numBlocks) {
        target += c;
        //IDX2C(i,j,ld)
        A[IDX2C(c, 0, lda)] = target[0];
        A[IDX2C(c, 1, lda)] = target[1];
        A[IDX2C(c, 2, lda)] = target[2];
        A[IDX2C(c, 3, lda)] = A[IDX2C(c, 4, lda)] = A[IDX2C(c, 5, lda)] = A[IDX2C(c, 6, lda)] = A[IDX2C(c, 7, lda)] = A[IDX2C(c, 8, lda)] = A[IDX2C(c, 10, lda)] = A[IDX2C(c, 11, lda)] = 0.0f;
        A[IDX2C(c, 9, lda)] = 1.0f;

        A[IDX2C((c + 1), 3, lda)] = target[0];
        A[IDX2C((c + 1), 4, lda)] = target[1];
        A[IDX2C((c + 1), 5, lda)] = target[2];
        A[IDX2C((c + 1), 0, lda)] = A[IDX2C((c + 1), 1, lda)] = A[IDX2C((c + 1), 2, lda)] = A[IDX2C((c + 1), 6, lda)] = A[IDX2C((c + 1), 7, lda)] = A[IDX2C((c + 1), 8, lda)] = A[IDX2C((c + 1), 9, lda)] = A[IDX2C((c + 1), 11, lda)] = 0.0f;
        A[IDX2C((c + 1), 10, lda)] = 1.0f;

        A[IDX2C((c + 2), 6, lda)] = target[0];
        A[IDX2C((c + 2), 7, lda)] = target[1];
        A[IDX2C((c + 2), 8, lda)] = target[2];
        A[IDX2C((c + 2), 0, lda)] = A[IDX2C((c + 2), 1, lda)] = A[IDX2C((c + 2), 2, lda)] = A[IDX2C((c + 2), 3, lda)] = A[IDX2C((c + 2), 4, lda)] = A[IDX2C((c + 2), 5, lda)] = A[IDX2C((c + 2), 9, lda)] = A[IDX2C((c + 2), 10, lda)] = 0.0f;
        A[IDX2C((c + 2), 11, lda)] = 1.0f;
    }
}
/* *************************************************************** */
//launched as ldm blocks n threads
__global__ void scaleV(float* V, const unsigned int ldm, const unsigned int n, float*w)
{
    unsigned int k = blockIdx.x;
    unsigned int j = threadIdx.x;
    V[IDX2C(j, k, ldm)] *= w[j];
}
/* *************************************************************** */
//launched as 1 block 1 thread
__global__ void outputMatFlat(float* mat, const unsigned int ldm, const unsigned int n, char* msg)
{
    for (int i = 0; i < ldm * n; ++i)
        printf("%f | ", mat[i]);
    printf("\n");
}
/* *************************************************************** */
//launched as 1 block 1 thread
__global__ void outputMat(float* mat, const unsigned int ldm, const unsigned int n, char* msg)
{
    for (int i = 0; i < ldm; ++i) {
        printf("%d ", i);
        for (int j = 0; j < n; ++j) {
            printf("%f ", mat[IDX2C(i, j, ldm)]);
        }
        printf("\n");
    }
    printf("\n");
}
/* *************************************************************** */
//blocks: 1 | threads: 12
__global__ void trimAndInvertSingularValuesKernel(float* sigma)
{
    sigma[threadIdx.x] = (sigma[threadIdx.x] < 0.0001) ? 0.0f : (1.0 / sigma[threadIdx.x]);
}
/* *************************************************************** */
__device__ void reg_mat44_dispCmat(float *mat, char * title, int tid)
{
    if (tid == 0)
        printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", title,
        mat[0 * 4 + 0], mat[0 * 4 + 1], mat[0 * 4 + 2], mat[0 * 4 + 3],
        mat[1 * 4 + 0], mat[1 * 4 + 1], mat[1 * 4 + 2], mat[1 * 4 + 3],
        mat[2 * 4 + 0], mat[2 * 4 + 1], mat[2 * 4 + 2], mat[2 * 4 + 3],
        mat[3 * 4 + 0], mat[3 * 4 + 1], mat[3 * 4 + 2], mat[3 * 4 + 3]);
}
/* *************************************************************** */
//threads: 16 | blocks:1
__global__ void permuteAffineMatrix(float* transform)
{
    __shared__ float buffer[16];
    const unsigned int i = threadIdx.x;

    buffer[i] = transform[i];
    __syncthreads();
    const unsigned int idx33 = (i / 3) * 4 + i % 3;
    const unsigned int idx34 = (i % 3) * 4 + 3;

    if (i < 9) transform[idx33] = buffer[i];
    else if (i < 12)transform[idx34] = buffer[i];
    else transform[i] = buffer[i];

}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void transformResultPointsKernel(float* transform, float* in, float* out, unsigned int definedBlockNum)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < definedBlockNum) {
        const unsigned int posIdx = 3 * tid;
        in += posIdx;
        out += posIdx;
        reg_mat44_mul_cuda<float>(transform, in, out);
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
#endif
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	NR_CUDA_SAFE_CALL(cudaMemcpy((void * )definedBlock_h, (void * )definedBlock_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	params->definedActiveBlockNumber = *definedBlock_h;
    printf("kernel definedActiveBlock: %d\n", params->definedActiveBlockNumber);
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(totalBlock_texture));

	free(definedBlock_h);
	cudaFree(definedBlock_d);

}

/* *************************************************************** */
//enable when cuda 7 is available?
#ifdef CUDA7
void checkCublasStatus(cublasStatus_t status)
{
	if (status != CUBLAS_STATUS_SUCCESS) {
		reg_print_fct_error("checkCublasStatus");
		reg_print_msg_error("!!!! CUBLAS  error");
		reg_exit(0);
	}
}
/* *************************************************************** */
void checkCUSOLVERStatus(cusolverStatus_t status, char* msg) {

	if (status != CUSOLVER_STATUS_SUCCESS) {
		if (status == CUSOLVER_STATUS_NOT_INITIALIZED)
			reg_print_fct_error("the library was not initialized.")
		else if (status == CUSOLVER_STATUS_INTERNAL_ERROR)
			reg_print_fct_error(" an internal operation failed.");

		reg_exit(0);
	}
}
/* *************************************************************** */
void checkDevInfo(int *devInfo) {
	int * hostDevInfo = (int*) malloc(sizeof(int));
	cudaMemcpy(hostDevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (hostDevInfo < 0)
		printf("parameter: %d is wrong\n", hostDevInfo);
	if (hostDevInfo > 0)
		printf("%d superdiagonals of an intermediate bidiagonal form B did not converge to zero.\n", hostDevInfo);
	else
		printf(" %d: operation successful\n", hostDevInfo);
	free(hostDevInfo);
}
/* *************************************************************** */
void downloadMat44(mat44 *lastTransformation, float* transform_d) {
	float* tempMat = (float*) malloc(16 * sizeof(float));
	cudaMemcpy(tempMat, transform_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	cPtrToMat44(lastTransformation, tempMat);
	free(tempMat);
}
/* *************************************************************** */
void uploadMat44(mat44 lastTransformation, float* transform_d) {
	float* tempMat = (float*) malloc(16 * sizeof(float));
	mat44ToCptr(lastTransformation, tempMat);
	cudaMemcpy(transform_d, tempMat, 16 * sizeof(float), cudaMemcpyHostToDevice);
	free(tempMat);
}
/* *************************************************************** */
/*
 * the function computes the SVD of a matrix A
 * A = V* x S x U, where V* is a (conjugate) transpose of V
 * */
void cusolverSVD(float* A_d, unsigned int m, unsigned int n, float* S_d, float* VT_d, float* U_d) {

	const int lda = m;
	const int ldu = m;
	const int ldvt = n;

	/*
	 * 'A': all m columns of U are returned in array
	 * 'S': the first min(m,n) columns of U (the left singular vectors) are returned in the array
	 * 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on the array
	 * 'N': no columns of U (no left singular vectors) are computed
	 */
	const char jobu = 'A';

	/*
	 * 'A': all N rows of V**T are returned in the array
	 * 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the array
	 * 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array
	 * 'N': no rows of V**T (no right singular vectors) are computed
	 */
	const char jobvt = 'A';

	cusolverDnHandle_t gH = NULL;
	int Lwork;
	//device ptrs
	float *Work;
	float *rwork;
	int *devInfo;

	//init cusolver compute SVD and shut down
	checkCUSOLVERStatus(cusolverDnCreate(&gH), "cusolverDnCreate");
	checkCUSOLVERStatus(cusolverDnSgesvd_bufferSize(gH, m, n, &Lwork), "cusolverDnSgesvd_bufferSize");

	cudaMalloc(&Work, Lwork * sizeof(float));
	cudaMalloc(&rwork, Lwork * sizeof(float));
	cudaMalloc(&devInfo, sizeof(int));

	checkCUSOLVERStatus(cusolverDnSgesvd(gH, jobu, jobvt, m, n, A_d, lda, S_d, U_d, ldu, VT_d, ldvt, Work, Lwork, NULL, devInfo), "cusolverDnSgesvd");
	checkCUSOLVERStatus(cusolverDnDestroy(gH), "cusolverDnDestroy");

	//free vars
	cudaFree(devInfo);
	cudaFree(rwork);
	cudaFree(Work);

}
/* *************************************************************** */
/*
 * the function computes the Pseudoinverse from the products of the SVD factorisation of A
 * R = V x inv(S) x U*
 * */
void cublasPseudoInverse(float* transformation, float *R_d, float* result_d, float *VT_d, float* Sigma_d, float *U_d, const unsigned int m, const unsigned int n) {
	// First we make sure that the really small singular values
	// are set to 0. and compute the inverse by taking the reciprocal of the entries

	trimAndInvertSingularValuesKernel<<<1, n>>>(Sigma_d);	//test 3

	cublasHandle_t handle;

	const float alpha = 1.f;
	const float beta = 0.f;

	const int ldvt = n;//VT's lead dimension
	const int ldu = m;//U's lead dimension
	const int ldr = n;//Pseudoinverse's r lead dimension

	const int rowsVTandR = n;//VT and r's num rows
	const int colsUandR = m;//U and r's num cols
	const int colsVtRowsU = n;//VT's cols and U's rows

	// V x inv(S) in place | We scale eaach row with the corresponding singular value as V is transpose
	scaleV<<<n,n>>>(VT_d, n, n, Sigma_d);

	//Initialize CUBLAS perform ops and shut down
	checkCublasStatus(cublasCreate(&handle));

	//now R = V x inv(S) x U*
	checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, rowsVTandR, colsUandR, colsVtRowsU, &alpha, VT_d, ldvt, U_d, ldu, &beta, R_d, ldr));

	//finally M=Rxb, where M is our affine matrix and b a vector containg the result points
	checkCublasStatus(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, R_d, ldr, result_d, 1, &beta, transformation, 1));
	checkCublasStatus(cublasDestroy(handle));
	permuteAffineMatrix<<<1,16>>>(transformation);
	cudaThreadSynchronize();

}

/* *************************************************************** */
//OPTIMIZER-----------------------------------------------

// estimate an affine transformation using least square
void getAffineMat3D(float* AR_d, float* Sigma_d, float* VT_d, float* U_d, float* target_d, float* result_d, float *transformation, const unsigned int numBlocks, unsigned int m, unsigned int n) {

	//populate A
	populateMatrixA<<<numBlocks, 512>>>(AR_d,target_d, m/3); //test 2

	//calculate SVD on the GPU
	cusolverSVD(AR_d, m, n, Sigma_d, VT_d, U_d);
	//calculate the pseudoinverse
	cublasPseudoInverse(transformation, AR_d,result_d, VT_d,Sigma_d, U_d, m, n);

}
/* *************************************************************** */

void optimize_affine3D_cuda(mat44 *cpuMat, float* final_d, float* AR_d, float* U_d, float* Sigma_d, float* VT_d, float* lengths_d, float* target_d, float* result_d, float* newResult_d, unsigned int m, unsigned int n, const unsigned int numToKeep, bool ilsIn) {

	//m | blockMatchingParams->definedActiveBlock * 3
	//n | 12
	const unsigned int numEquations = m / 3;
	const unsigned int numBlocks = (numEquations % 512) ? (numEquations / 512) + 1 : numEquations / 512;

	uploadMat44(*cpuMat, final_d);
	transformResultPointsKernel<<<numBlocks, 512>>>(final_d, result_d,newResult_d, m/3); //test 1
	cudaMemcpy(result_d, newResult_d, m * sizeof(float), cudaMemcpyDeviceToDevice);

	// run the local search optimization routine
	affineLocalSearch3DCuda(cpuMat, final_d, AR_d, Sigma_d, U_d, VT_d, newResult_d, target_d, result_d, lengths_d, numBlocks, numToKeep, m, n);

	downloadMat44(cpuMat, final_d);
}
/* *************************************************************** */
void affineLocalSearch3DCuda(mat44 *cpuMat, float* final_d, float *AR_d, float* Sigma_d, float* U_d, float* VT_d, float * newResultPos_d, float* targetPos_d, float* resultPos_d, float* lengths_d, const unsigned int numBlocks, const unsigned int num_to_keep, const unsigned int m, const unsigned int n) {

	double lastDistance = std::numeric_limits<double>::max();

	float* lastTransformation_d;
	cudaMalloc(&lastTransformation_d, 16 * sizeof(float));

	//get initial affine matrix
	getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, final_d, numBlocks, m, n);

	for (unsigned int count = 0; count < MAX_ITERATIONS; ++count) {

		// Transform the points in the target
		transformResultPointsKernel<<<numBlocks, 512>>>(final_d, targetPos_d,newResultPos_d, m/3); //test 1
		double distance = sortAndReduce( lengths_d, targetPos_d, resultPos_d, newResultPos_d, numBlocks,num_to_keep, m);

		// If the change is not substantial or we are getting worst, we return
		if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE) break;

		lastDistance = distance;

		cudaMemcpy( lastTransformation_d,final_d, 16*sizeof(float), cudaMemcpyDeviceToDevice);
		getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, final_d, numBlocks, m, n);
	}

	//async cudamemcpy here
	cudaMemcpy(final_d, lastTransformation_d, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(lastTransformation_d);
}
#endif //IF CUDA7
#endif //_REG_BLOCKMATCHING_GPU_CU
