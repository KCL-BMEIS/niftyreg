/*
 *  _reg_blockMatching_kernels.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_KERNELS_CU__
#define __REG_BLOCKMATCHING_KERNELS_CU__

#include "assert.h"
#include "_reg_blockMatching.h"

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

    const unsigned int idy = threadIdx.x/4;
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

            const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float) 0 };

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
#endif
