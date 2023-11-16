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
namespace NiftyReg::Cuda {
/* *************************************************************** */
__global__ void GetSsdValueKernel(float *ssdSum,
                                  float *ssdCount,
                                  cudaTextureObject_t referenceTexture,
                                  cudaTextureObject_t warpedTexture,
                                  cudaTextureObject_t localWeightSimTexture,
                                  cudaTextureObject_t maskTexture,
                                  const int3 referenceImageDim,
                                  const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);

        const float warValue = tex1Dfetch<float>(warpedTexture, index);
        if (warValue != warValue) return;

        const auto&& [x, y, z] = reg_indexToDims_cuda(index, referenceImageDim);
        const float refValue = tex3D<float>(referenceTexture, x, y, z);
        if (refValue != refValue) return;

        const float val = localWeightSimTexture ? tex1Dfetch<float>(localWeightSimTexture, index) : 1.f;
        const float diff = refValue - warValue;
        atomicAdd(ssdSum, diff * diff * val);
        atomicAdd(ssdCount, val);
    }
}
/* *************************************************************** */
__global__ void GetSsdGradientKernel(float4 *ssdGradient,
                                     cudaTextureObject_t referenceTexture,
                                     cudaTextureObject_t warpedTexture,
                                     cudaTextureObject_t maskTexture,
                                     cudaTextureObject_t spatialGradTexture,
                                     cudaTextureObject_t localWeightSimTexture,
                                     const int3 referenceImageDim,
                                     const float adjustedWeight,
                                     const unsigned activeVoxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int index = tex1Dfetch<int>(maskTexture, tid);

        const float warValue = tex1Dfetch<float>(warpedTexture, index);
        if (warValue != warValue) return;

        const float4 spaGradientValue = tex1Dfetch<float4>(spatialGradTexture, tid);
        if (spaGradientValue.x != spaGradientValue.x ||
            spaGradientValue.y != spaGradientValue.y ||
            spaGradientValue.z != spaGradientValue.z)
            return;

        const auto&& [x, y, z] = reg_indexToDims_cuda(index, referenceImageDim);
        const float refValue = tex3D<float>(referenceTexture, x, y, z);
        if (refValue != refValue) return;

        const float val = localWeightSimTexture ? tex1Dfetch<float>(localWeightSimTexture, index) : 1.f;
        const float common = -2.f * (refValue - warValue) * adjustedWeight * val;

        float4 ssdGradientValue = ssdGradient[index];
        ssdGradientValue.x += common * spaGradientValue.x;
        ssdGradientValue.y += common * spaGradientValue.y;
        ssdGradientValue.z += common * spaGradientValue.z;
        ssdGradient[index] = ssdGradientValue;
    }
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
