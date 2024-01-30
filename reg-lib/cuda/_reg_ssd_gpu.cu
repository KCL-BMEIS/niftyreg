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
                                    nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda) {
    reg_ssd::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda, floMask, floMaskCuda, warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda);
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
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    Cuda::UniqueTextureObjectPtr localWeightSimTexturePtr; cudaTextureObject_t localWeightSimTexture = 0;
    if (localWeightSimCuda) {
        localWeightSimTexturePtr = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
        localWeightSimTexture = *localWeightSimTexturePtr;
    }

    double ssd = 0.0;
    for (int t = 0; t < referenceTimePoints; t++) {
        auto referenceTexturePtr = Cuda::CreateTextureObject(referenceImageCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        auto warpedTexturePtr = Cuda::CreateTextureObject(warpedCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        auto referenceTexture = *referenceTexturePtr;
        auto warpedTexture = *warpedTexturePtr;

        const auto ssdAndCount = thrust::transform_reduce(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int index) -> double2 {
            const double refValue = tex1Dfetch<float>(referenceTexture, index);
            if (refValue != refValue) return {};

            const double warValue = tex1Dfetch<float>(warpedTexture, index);
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
                               this->activeVoxelNumber,
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
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexturePtr = Cuda::CreateTextureObject(referenceImageCuda + currentTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto warpedTexturePtr = Cuda::CreateTextureObject(warpedCuda + currentTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto spatialGradTexturePtr = Cuda::CreateTextureObject(spatialGradCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto referenceTexture = *referenceTexturePtr;
    auto warpedTexture = *warpedTexturePtr;
    auto spatialGradTexture = *spatialGradTexturePtr;
    Cuda::UniqueTextureObjectPtr localWeightSimTexturePtr; cudaTextureObject_t localWeightSimTexture = 0;
    if (localWeightSimCuda) {
        localWeightSimTexturePtr = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
        localWeightSimTexture = *localWeightSimTexturePtr;
    }

    // Find number of valid voxels and correct weight
    const auto validVoxelNumber = thrust::count_if(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int index) {
        const float refValue = tex1Dfetch<float>(referenceTexture, index);
        if (refValue != refValue) return false;
        const float warValue = tex1Dfetch<float>(warpedTexture, index);
        if (warValue != warValue) return false;
        return true;
    });
    const double adjustedWeight = timePointWeight / validVoxelNumber;

    // Calculate the SSD gradient
    thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
        const double refValue = tex1Dfetch<float>(referenceTexture, index);
        if (refValue != refValue) return;

        const double warValue = tex1Dfetch<float>(warpedTexture, index);
        if (warValue != warValue) return;

        const float4 spaGradientValue = tex1Dfetch<float4>(spatialGradTexture, index);
        if (spaGradientValue.x != spaGradientValue.x ||
            spaGradientValue.y != spaGradientValue.y ||
            spaGradientValue.z != spaGradientValue.z)
            return;

        const double weight = localWeightSimTexture ? tex1Dfetch<float>(localWeightSimTexture, index) : 1.f;
        const double common = -2.0 * (refValue - warValue) * adjustedWeight * weight;

        float4 ssdGradientValue = ssdGradientCuda[index];
        ssdGradientValue.x += common * spaGradientValue.x;
        ssdGradientValue.y += common * spaGradientValue.y;
        ssdGradientValue.z += common * spaGradientValue.z;
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
                                     this->activeVoxelNumber,
                                     this->timePointWeights[currentTimePoint],
                                     currentTimePoint);
}
/* *************************************************************** */
