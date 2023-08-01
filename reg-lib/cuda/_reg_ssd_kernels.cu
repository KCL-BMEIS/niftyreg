/*
 * @file _reg_ssd_kernels.cu
 * @author Marc Modat
 * @date 14/11/2012
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_ssd_gpu.h"
#include "_reg_ssd_kernels.cu"
#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
__global__ void reg_getSquaredDifference3d_kernel(float *squaredDifference,
                                                  cudaTextureObject_t referenceTexture,
                                                  cudaTextureObject_t warpedTexture,
                                                  cudaTextureObject_t maskTexture,
                                                  const int3 referenceImageDim,
                                                  const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);
        int quot, rem;
        reg_div_cuda(index, referenceImageDim.x * referenceImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float difference = tex3D<float>(referenceTexture,
                                        ((float)x + 0.5f) / (float)referenceImageDim.x,
                                        ((float)y + 0.5f) / (float)referenceImageDim.y,
                                        ((float)z + 0.5f) / (float)referenceImageDim.z);
        difference -= tex1Dfetch<float>(warpedTexture, index);
        squaredDifference[tid] = difference == difference ? difference * difference : 0;
    }
}
/* *************************************************************** */
__global__ void reg_getSquaredDifference2d_kernel(float *squaredDifference,
                                                  cudaTextureObject_t referenceTexture,
                                                  cudaTextureObject_t warpedTexture,
                                                  cudaTextureObject_t maskTexture,
                                                  const int3 referenceImageDim,
                                                  const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);
        int quot, rem;
        reg_div_cuda(index, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float difference = tex3D<float>(referenceTexture,
                                        ((float)x + 0.5f) / (float)referenceImageDim.x,
                                        ((float)y + 0.5f) / (float)referenceImageDim.y,
                                        0.5f);
        difference -= tex1Dfetch<float>(warpedTexture, index);
        squaredDifference[tid] = difference == difference ? difference * difference : 0;
    }
}
/* *************************************************************** */
__global__ void reg_getSsdGradient2d_kernel(float4 *ssdGradient,
                                            cudaTextureObject_t referenceTexture,
                                            cudaTextureObject_t warpedTexture,
                                            cudaTextureObject_t maskTexture,
                                            cudaTextureObject_t spaGradientTexture,
                                            const int3 referenceImageDim,
                                            const float maxSD,
                                            const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);
        int quot, rem;
        reg_div_cuda(index, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        const float refValue = tex3D<float>(referenceTexture,
                                            ((float)x + 0.5f) / (float)referenceImageDim.x,
                                            ((float)y + 0.5f) / (float)referenceImageDim.y,
                                            0.5f);
        if (refValue != refValue)
            return;
        const float warpValue = tex1Dfetch<float>(warpedTexture, index);
        if (warpValue != warpValue)
            return;

        const float4 spaGradientValue = tex1Dfetch<float4>(spaGradientTexture, tid);
        if (spaGradientValue.x != spaGradientValue.x || spaGradientValue.y != spaGradientValue.y)
            return;

        const float common = -2.f * (refValue - warpValue) / (maxSD * (float)activeVoxelNumber);
        ssdGradient[index] = make_float4(common * spaGradientValue.x, common * spaGradientValue.y, 0.f, 0.f);
    }
}
/* *************************************************************** */
__global__ void reg_getSsdGradient3d_kernel(float4 *ssdGradient,
                                            cudaTextureObject_t referenceTexture,
                                            cudaTextureObject_t warpedTexture,
                                            cudaTextureObject_t maskTexture,
                                            cudaTextureObject_t spaGradientTexture,
                                            const int3 referenceImageDim,
                                            const float maxSD,
                                            const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);
        int quot, rem;
        reg_div_cuda(index, referenceImageDim.x * referenceImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        const float refValue = tex3D<float>(referenceTexture,
                                            ((float)x + 0.5f) / (float)referenceImageDim.x,
                                            ((float)y + 0.5f) / (float)referenceImageDim.y,
                                            ((float)z + 0.5f) / (float)referenceImageDim.z);
        if (refValue != refValue)
            return;

        const float warpValue = tex1Dfetch<float>(warpedTexture, index);
        if (warpValue != warpValue)
            return;

        const float4 spaGradientValue = tex1Dfetch<float4>(spaGradientTexture, tid);
        if (spaGradientValue.x != spaGradientValue.x ||
            spaGradientValue.y != spaGradientValue.y ||
            spaGradientValue.z != spaGradientValue.z)
            return;

        const float common = -2.f * (refValue - warpValue) / (maxSD * (float)activeVoxelNumber);
        ssdGradient[index] = make_float4(common * spaGradientValue.x, common * spaGradientValue.y, common * spaGradientValue.z, 0.f);
    }
}
/* *************************************************************** */
