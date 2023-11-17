/*
 *  blockMatchingKernel.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *
 */

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
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
// Some parameters that we need for the kernel execution.
// The caller is supposed to ensure that the values are set

// Transformation matrix from nifti header
__device__ __constant__ float4 t_m_a;
__device__ __constant__ float4 t_m_b;
__device__ __constant__ float4 t_m_c;

#define BLOCK_WIDTH   4
#define BLOCK_SIZE    64
#define OVERLAP_SIZE  3
#define STEP_SIZE     1

/* *************************************************************** */
template<class DataType>
__device__ __inline__ void reg2D_mat44_mul_cuda(const float *mat, const DataType *in, DataType *out) {
    out[0] = (DataType)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * 0 + (double)mat[0 * 4 + 3]);
    out[1] = (DataType)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * 0 + (double)mat[1 * 4 + 3]);
}
template<class DataType>
__device__ __inline__ void reg_mat44_mul_cuda(const float *mat, const DataType *in, DataType *out) {
    out[0] = (DataType)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
    out[1] = (DataType)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
    out[2] = (DataType)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
}
// Apply the transformation matrix
__device__ __inline__ void apply_affine(const float4& pt, float *result) {
    float4 mat = t_m_a;
    result[0] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
    mat = t_m_b;
    result[1] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
    mat = t_m_c;
    result[2] = (mat.x * pt.x) + (mat.y * pt.y) + (mat.z * pt.z) + (mat.w);
}
/* *************************************************************** */
__device__ __inline__ float blockReduce2DSum(float val, unsigned tid) {
    static __shared__ float shared[16];
    __syncthreads();
    shared[tid] = val;
    __syncthreads();

    for (unsigned i = 8; i > 0; i >>= 1) {
        if (tid < i)
            shared[tid] += shared[tid + i];
        __syncthreads();
    }
    return shared[0];
}
/* *************************************************************** */
__device__ __inline__ float blockReduceSum(float val, unsigned tid) {
    static __shared__ float shared[64];
    __syncthreads();
    shared[tid] = val;
    __syncthreads();

    for (unsigned i = 32; i > 0; i >>= 1) {
        if (tid < i)
            shared[tid] += shared[tid + i];
        __syncthreads();
    }
    // if (tid == 0){
    //     for (unsigned i = 1; i < 64; ++i) {
    //             shared[0] += shared[i];
    //     }
    // }
    // __syncthreads();
    return shared[0];
}
/* *************************************************************** */
__global__ void blockMatchingKernel2D(float *warpedPosition,
                                      float *referencePosition,
                                      cudaTextureObject_t referenceTexture,
                                      cudaTextureObject_t warpedTexture,
                                      cudaTextureObject_t totalBlockTexture,
                                      const int *mask,
                                      const float* referenceMatrix,
                                      unsigned *definedBlock,
                                      const int3 imageSize,
                                      const uint3 blockSize) {
    extern __shared__ float sWarpedValues[];
    // Compute the current block index
    const unsigned bid = blockIdx.y * gridDim.x + blockIdx.x;

    const int currentBlockIndex = tex1Dfetch<int>(totalBlockTexture, bid);
    if (currentBlockIndex > -1) {
        const unsigned idy = threadIdx.x;
        const unsigned idx = threadIdx.y;
        const unsigned tid = idy * 4 + idx;
        const unsigned xImage = blockIdx.x * 4 + idx;
        const unsigned yImage = blockIdx.y * 4 + idy;

        //populate shared memory with resultImageArray's values
        for (int y = -1; y < 2; ++y) {
            const int yImageIn = yImage + y * 4;
            for (int x = -1; x < 2; ++x) {
                const int xImageIn = xImage + x * 4;
                const int sharedIndex = ((y + 1) * 4 + idy) * 12 + (x + 1) * 4 + idx;
                const int indexXYIn = yImageIn * imageSize.x + xImageIn;
                const bool valid =
                    (xImageIn > -1 && xImageIn < (int)imageSize.x) &&
                    (yImageIn > -1 && yImageIn < (int)imageSize.y);
                sWarpedValues[sharedIndex] = valid  ?
                    tex1Dfetch<float>(warpedTexture, indexXYIn) : nanf("sNaN");
            }
        }

        //for most cases we need this out of th loop
        //value if the block is 4x4 NaN otherwise
        const unsigned long voxIndex = yImage * imageSize.x + xImage;
        const bool referenceInBounds = xImage < imageSize.x && yImage < imageSize.y;
        float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
            tex1Dfetch<float>(referenceTexture, voxIndex) : nanf("sNaN");
        const bool finiteReference = isfinite(rReferenceValue);
        rReferenceValue = finiteReference ? rReferenceValue : 0.f;
        const unsigned referenceSize = __syncthreads_count(finiteReference);

        float bestDisplacement[2] = { nanf("sNaN"), 0.0f };
        float bestCC = 0;

        if (referenceSize > 8) {
            //the target values must remain constant throughout the block matching process
            const float referenceMean = __fdividef(blockReduce2DSum(rReferenceValue, tid), referenceSize);
            const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.f;
            const float referenceVar = blockReduce2DSum(referenceTemp * referenceTemp, tid);
            // iteration over the result blocks (block matching part)
            for (unsigned y = 1; y < 8; ++y) {
                for (unsigned x = 1; x < 8; ++x) {
                    const unsigned sharedIndex = (y + idy) * 12 + x + idx;
                    const float rWarpedValue = sWarpedValues[sharedIndex];
                    const bool overlap = isfinite(rWarpedValue) && finiteReference;
                    const unsigned warpedSize = __syncthreads_count(overlap);

                    if (warpedSize > 8) {
                        //the reference values must remain intact at each loop, so please do not touch this!
                        float newreferenceTemp = referenceTemp;
                        float newreferenceVar = referenceVar;
                        if (warpedSize != referenceSize) {
                            const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
                            const float newReferenceMean = __fdividef(blockReduce2DSum(newReferenceValue, tid), warpedSize);
                            newreferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
                            newreferenceVar = blockReduce2DSum(newreferenceTemp * newreferenceTemp, tid);
                        }

                        const float rChecked = overlap ? rWarpedValue : 0.0f;
                        const float warpedMean = __fdividef(blockReduce2DSum(rChecked, tid), warpedSize);
                        const float warpedTemp = overlap ? rChecked - warpedMean : 0.0f;
                        const float warpedVar = blockReduce2DSum(warpedTemp * warpedTemp, tid);

                        const float sumTargetResult = blockReduce2DSum((newreferenceTemp) * (warpedTemp), tid);
                        const float localCC = (newreferenceVar * warpedVar) > 0 ? fabs((sumTargetResult) / sqrt(newreferenceVar * warpedVar)) : 0;

                        if (tid == 0 && localCC > bestCC) {
                            bestCC = localCC + 1.0e-7f;
                            bestDisplacement[0] = x - 4.f;
                            bestDisplacement[1] = y - 4.f;
                        }
                    }
                }
            }
        }

        if (tid == 0) {
            const unsigned posIdx = 2 * currentBlockIndex;
            const float referencePosition_temp[2] = { (float)xImage, (float)yImage };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];

            reg2D_mat44_mul_cuda<float>(referenceMatrix, referencePosition_temp, &referencePosition[posIdx]);
            reg2D_mat44_mul_cuda<float>(referenceMatrix, bestDisplacement, &warpedPosition[posIdx]);

            if (isfinite(bestDisplacement[0]))
                atomicAdd(definedBlock, 1);
        }
    }
}
/* *************************************************************** */
__global__ void blockMatchingKernel3D(float *warpedPosition,
                                      float *referencePosition,
                                      cudaTextureObject_t referenceTexture,
                                      cudaTextureObject_t warpedTexture,
                                      cudaTextureObject_t totalBlockTexture,
                                      const int *mask,
                                      const float* referenceMatrix,
                                      unsigned *definedBlock,
                                      const int3 imageSize,
                                      const uint3 blockSize) {
    extern __shared__ float sWarpedValues[];
    // Compute the current block index
    const unsigned bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;

    const int currentBlockIndex = tex1Dfetch<int>(totalBlockTexture, bid);
    if (currentBlockIndex > -1) {
        const unsigned idx = threadIdx.x;
        const unsigned idy = threadIdx.y;
        const unsigned idz = threadIdx.z;
        const unsigned tid = (idz * 4 + idy) * 4 + idx;
        const unsigned xImage = blockIdx.x * 4 + idx;
        const unsigned yImage = blockIdx.y * 4 + idy;
        const unsigned zImage = blockIdx.z * 4 + idz;

        //populate shared memory with resultImageArray's values
        for (int z = -1; z < 2; ++z) {
            const int zImageIn = zImage + z * 4;
            for (int y = -1; y < 2; ++y) {
                const int yImageIn = yImage + y * 4;
                for (int x = -1; x < 2; ++x) {
                    const int xImageIn = xImage + x * 4;
                    const int sharedIndex = (((z + 1) * 4 + idz) * 12 + (y + 1) * 4 + idy) * 12 + (x + 1) * 4 + idx;
                    const unsigned indexXYZIn = xImageIn + imageSize.x * (yImageIn + zImageIn * imageSize.y);
                    const bool valid =
                        (xImageIn > -1 && xImageIn < (int)imageSize.x) &&
                        (yImageIn > -1 && yImageIn < (int)imageSize.y) &&
                        (zImageIn > -1 && zImageIn < (int)imageSize.z);
                    sWarpedValues[sharedIndex] = valid ?
                        tex1Dfetch<float>(warpedTexture, indexXYZIn) : nanf("sNaN");     //for some reason the mask here creates probs
                }
            }
        }

        //for most cases we need this out of th loop
        //value if the block is 4x4x4 NaN otherwise
        const unsigned voxIndex = (zImage * imageSize.y + yImage) * imageSize.x + xImage;
        const bool referenceInBounds = xImage < imageSize.x && yImage < imageSize.y && zImage < imageSize.z;
        float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
            tex1Dfetch<float>(referenceTexture, voxIndex) : nanf("sNaN");
        const bool finiteReference = isfinite(rReferenceValue);
        rReferenceValue = finiteReference ? rReferenceValue : 0.f;
        const unsigned referenceSize = __syncthreads_count(finiteReference);

        float bestDisplacement[3] = { nanf("sNaN"), 0.0f, 0.0f };
        float bestCC = 0.0f;

        if (referenceSize > 32) {
            //the target values must remain constant throughout the block matching process
            // const float referenceMean = __fdividef(blockReduceSum(rReferenceValue, tid), referenceSize);
            const float referenceMean = blockReduceSum(rReferenceValue, tid) / referenceSize;
            const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.f;
            const float referenceVar = blockReduceSum(referenceTemp * referenceTemp, tid);

            // iteration over the result blocks (block matching part)
            for (unsigned z = 1; z < 8; ++z) {
                for (unsigned y = 1; y < 8; ++y) {
                    for (unsigned x = 1; x < 8; ++x) {
                        const unsigned sharedIndex = ((z + idz) * 12 + y + idy) * 12 + x + idx;
                        const float rWarpedValue = sWarpedValues[sharedIndex];
                        const bool overlap = isfinite(rWarpedValue) && finiteReference;
                        const unsigned warpedSize = __syncthreads_count(overlap);

                        if (warpedSize > 32) {
                            //the target values must remain intact at each loop, so please do not touch this!
                            float newreferenceTemp = referenceTemp;
                            float newreferenceVar = referenceVar;
                            if (warpedSize != referenceSize) {
                                const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
                                // const float newReferenceMean = __fdividef(blockReduceSum(newReferenceValue, tid), warpedSize);
                                const float newReferenceMean = blockReduceSum(newReferenceValue, tid) / warpedSize;
                                newreferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
                                newreferenceVar = blockReduceSum(newreferenceTemp * newreferenceTemp, tid);
                            }

                            const float rChecked = overlap ? rWarpedValue : 0.0f;
                            // const float warpedMean = __fdividef(blockReduceSum(rChecked, tid), warpedSize);
                            const float warpedMean = blockReduceSum(rChecked, tid) / warpedSize;
                            const float warpedTemp = overlap ? rChecked - warpedMean : 0.0f;
                            const float warpedVar = blockReduceSum(warpedTemp * warpedTemp, tid);

                            const float sumTargetResult = blockReduceSum(newreferenceTemp * warpedTemp, tid);
                            const float localCC = (newreferenceVar * warpedVar) > 0 ? fabs(
                                sumTargetResult / sqrt(newreferenceVar * warpedVar)) : 0;

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

        if (tid == 0) {
            const unsigned posIdx = 3 * currentBlockIndex;
            const float referencePosition_temp[3] = { (float)xImage, (float)yImage, (float)zImage };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] += referencePosition_temp[2];

            reg_mat44_mul_cuda<float>(referenceMatrix, referencePosition_temp, &referencePosition[posIdx]);
            reg_mat44_mul_cuda<float>(referenceMatrix, bestDisplacement, &warpedPosition[posIdx]);
            if (isfinite(bestDisplacement[0]))
                atomicAdd(definedBlock, 1);
        }
    }
}
/* *************************************************************** */
void block_matching_method_gpu(const nifti_image *referenceImage,
                               _reg_blockMatchingParam *params,
                               const float *referenceImageCuda,
                               const float *warpedImageCuda,
                               float *referencePositionCuda,
                               float *warpedPositionCuda,
                               const int *totalBlockCuda,
                               const int *maskCuda,
                               const float *refMatCuda) {
    if (params->stepSize != 1 || params->voxelCaptureRange != 3)
        NR_FATAL_ERROR("The block matching CUDA kernel supports only single step size!");

    const int3 imageSize = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const uint3 blockSize = make_uint3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
    const unsigned numBlocks = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];

    auto referenceTexture = Cuda::CreateTextureObject(referenceImageCuda, referenceImage->nvox, cudaChannelFormatKindFloat, 1);
    auto warpedTexture = Cuda::CreateTextureObject(warpedImageCuda, referenceImage->nvox, cudaChannelFormatKindFloat, 1);
    auto totalBlockTexture = Cuda::CreateTextureObject(totalBlockCuda, numBlocks, cudaChannelFormatKindSigned, 1);

    unsigned definedBlock = 0, *definedBlockCuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&definedBlockCuda, sizeof(unsigned)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(definedBlockCuda, &definedBlock, sizeof(unsigned), cudaMemcpyHostToDevice));

    dim3 blockDims(4, 4, 4);
    dim3 gridDims(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
    unsigned sharedMemSize = (64 + 4 * 3 * 4 * 3 * 4 * 3) * sizeof(float);  // (3*4)^3

    if (referenceImage->nz == 1) {
        blockDims.z = 1;
        gridDims.z = 1;
        sharedMemSize = (16 + 144) * sizeof(float);  // (3*4)^2
        blockMatchingKernel2D<<<gridDims, blockDims, sharedMemSize>>>(warpedPositionCuda,
                                                                      referencePositionCuda,
                                                                      *referenceTexture,
                                                                      *warpedTexture,
                                                                      *totalBlockTexture,
                                                                      maskCuda,
                                                                      refMatCuda,
                                                                      definedBlockCuda,
                                                                      imageSize,
                                                                      blockSize);
    } else {
        blockMatchingKernel3D<<<gridDims, blockDims, sharedMemSize>>>(warpedPositionCuda,
                                                                      referencePositionCuda,
                                                                      *referenceTexture,
                                                                      *warpedTexture,
                                                                      *totalBlockTexture,
                                                                      maskCuda,
                                                                      refMatCuda,
                                                                      definedBlockCuda,
                                                                      imageSize,
                                                                      blockSize);
    }
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    NR_CUDA_SAFE_CALL(cudaMemcpy(&definedBlock, definedBlockCuda, sizeof(unsigned), cudaMemcpyDeviceToHost));
    params->definedActiveBlockNumber = definedBlock;
    NR_CUDA_SAFE_CALL(cudaFree(definedBlockCuda));
}
/* *************************************************************** */
