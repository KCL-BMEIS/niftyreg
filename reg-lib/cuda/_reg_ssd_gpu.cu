/*
 * @file _reg_ssd_gpu.cu
 * @author Marc Modat
 * @date 14/11/2012
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ssd_gpu.h"

/* *************************************************************** */
reg_ssd_gpu::reg_ssd_gpu(): reg_ssd::reg_ssd() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_ssd_gpu::~reg_ssd_gpu() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_ssd_gpu::InitialiseMeasure(nifti_image *refImg, float *refImgCuda,
                                    nifti_image *floImg, float *floImgCuda,
                                    int *refMask, int *refMaskCuda,
                                    size_t activeVoxNum,
                                    nifti_image *warpedImg, float *warpedImgCuda,
                                    nifti_image *warpedGrad, float4 *warpedGradCuda,
                                    nifti_image *voxelBasedGrad, float4 *voxelBasedGradCuda,
                                    nifti_image *localWeightSim, float *localWeightSimCuda,
                                    int *floMask, int *floMaskCuda,
                                    nifti_image *warpedImgBw, float *warpedImgBwCuda,
                                    nifti_image *warpedGradBw, float4 *warpedGradBwCuda,
                                    nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda,
                                    size_t activeVoxNumBw) {
    reg_ssd::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda, floMask, floMaskCuda, warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda, activeVoxNumBw);
    // Check if the reference and floating images need to be updated
    for (int i = 0; i < this->referenceTimePoints; ++i)
        if (this->timePointWeights[i] > 0 && normaliseTimePoint[i]) {
            Cuda::TransferNiftiToDevice(this->referenceImageCuda, this->referenceImage);
            Cuda::TransferNiftiToDevice(this->floatingImageCuda, this->floatingImage);
            break;
        }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
double reg_getSsdValue_gpu(const nifti_image *referenceImage,
                           const float *referenceImageCuda,
                           const float *warpedCuda,
                           const float *localWeightSimCuda,
                           const int *maskCuda,
                           const size_t activeVoxelNumber,
                           const double *timePointWeights,
                           const int referenceTimePoints) {
    const int3 referenceImageDims = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    Cuda::UniqueTextureObjectPtr localWeightSimTexturePtr; cudaTextureObject_t localWeightSimTexture = 0;
    if (localWeightSimCuda) {
        localWeightSimTexturePtr = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
        localWeightSimTexture = *localWeightSimTexturePtr;
    }

    double ssd = 0.0;
    for (int t = 0; t < referenceTimePoints; t++) {
        // A time point starts at an offset that is not necessarily 512-byte aligned, so it cannot be
        // bound to a texture (cudaCreateTextureObject rejects unaligned addresses)
        const float *refPtr = referenceImageCuda + t * voxelNumber;
        const float *warPtr = warpedCuda + t * voxelNumber;

        const auto ssdAndCount = thrust::transform_reduce(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int index) -> double2 {
            const double refValue = refPtr[index];
            if (refValue != refValue) return {};

            const double warValue = warPtr[index];
            if (warValue != warValue) return {};

            const double weight = localWeightSimTexture ? tex1Dfetch<float>(localWeightSimTexture, index) : 1.f;
            const double diff = refValue - warValue;
            return { Square(diff) * weight, weight };  // ssd and count
        }, make_double2(0, 0), thrust::plus<double2>());

        ssd += (ssdAndCount.x * timePointWeights[t]) / ssdAndCount.y;
    }

    return -ssd;
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueFw() {
    return reg_getSsdValue_gpu(this->referenceImage,
                               this->referenceImageCuda,
                               this->warpedImageCuda,
                               this->localWeightSimCuda,
                               this->referenceMaskCuda,
                               this->activeVoxelNumber,
                               this->timePointWeights,
                               this->referenceTimePoints);
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueBw() {
    return reg_getSsdValue_gpu(this->floatingImage,
                               this->floatingImageCuda,
                               this->warpedImageBwCuda,
                               nullptr,
                               this->floatingMaskCuda,
                               this->activeVoxelNumberBw,
                               this->timePointWeights,
                               this->referenceTimePoints);
}
/* *************************************************************** */
void reg_getVoxelBasedSsdGradient_gpu(const nifti_image *referenceImage,
                                      const float *referenceImageCuda,
                                      const float *warpedCuda,
                                      const float4 *spatialGradCuda,
                                      const float *localWeightSimCuda,
                                      float4 *ssdGradientCuda,
                                      const int *maskCuda,
                                      const size_t activeVoxelNumber,
                                      const double timePointWeight,
                                      const int currentTimePoint) {
    const int3 referenceImageDims = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    // A time point starts at an offset that is not necessarily 512-byte aligned, so it cannot be
    // bound to a texture (cudaCreateTextureObject rejects unaligned addresses)
    const float *refPtr = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warPtr = warpedCuda + currentTimePoint * voxelNumber;
    auto spatialGradTexturePtr = Cuda::CreateTextureObject(spatialGradCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto spatialGradTexture = *spatialGradTexturePtr;
    Cuda::UniqueTextureObjectPtr localWeightSimTexturePtr; cudaTextureObject_t localWeightSimTexture = 0;
    if (localWeightSimCuda) {
        localWeightSimTexturePtr = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
        localWeightSimTexture = *localWeightSimTexturePtr;
    }

    // Find number of valid voxels and correct weight
    const auto validVoxelNumber = thrust::count_if(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int index) -> bool {
        const float refValue = refPtr[index];
        if (refValue != refValue) return false;
        const float warValue = warPtr[index];
        if (warValue != warValue) return false;
        return true;
    });
    const double adjustedWeight = timePointWeight / validVoxelNumber;

    // Calculate the SSD gradient
    thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
        const double refValue = refPtr[index];
        if (refValue != refValue) return;

        const double warValue = warPtr[index];
        if (warValue != warValue) return;

        const float4 spaGradientValue = tex1Dfetch<float4>(spatialGradTexture, index);
        if (spaGradientValue.x != spaGradientValue.x ||
            spaGradientValue.y != spaGradientValue.y ||
            spaGradientValue.z != spaGradientValue.z)
            return;

        const double weight = localWeightSimTexture ? tex1Dfetch<float>(localWeightSimTexture, index) : 1.f;
        // Match the CPU multiply order (weight before adjustedWeight) and round each product to
        // float before accumulating, mirroring reg_getVoxelBasedSsdGradient in _reg_ssd.cpp. This
        // keeps the CUDA gradient bit-exact with the CPU regardless of the FMA build flag.
        const double common = -2.0 * (refValue - warValue) * weight * adjustedWeight;

        float4 ssdGradientValue = ssdGradientCuda[index];
        ssdGradientValue.x += static_cast<float>(common * spaGradientValue.x);
        ssdGradientValue.y += static_cast<float>(common * spaGradientValue.y);
        ssdGradientValue.z += static_cast<float>(common * spaGradientValue.z);
        ssdGradientCuda[index] = ssdGradientValue;
    });
}
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    reg_getVoxelBasedSsdGradient_gpu(this->referenceImage,
                                     this->referenceImageCuda,
                                     this->warpedImageCuda,
                                     this->warpedGradientCuda,
                                     this->localWeightSimCuda,
                                     this->voxelBasedGradientCuda,
                                     this->referenceMaskCuda,
                                     this->activeVoxelNumber,
                                     this->timePointWeights[currentTimePoint],
                                     currentTimePoint);
}
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    reg_getVoxelBasedSsdGradient_gpu(this->floatingImage,
                                     this->floatingImageCuda,
                                     this->warpedImageBwCuda,
                                     this->warpedGradientBwCuda,
                                     nullptr,
                                     this->voxelBasedGradientBwCuda,
                                     this->floatingMaskCuda,
                                     this->activeVoxelNumberBw,
                                     this->timePointWeights[currentTimePoint],
                                     currentTimePoint);
}
/* *************************************************************** */
