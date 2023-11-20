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
#include "_reg_ssd_kernels.cu"

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
    // Check that the input images have only one time point
    if (this->referenceImage->nt > 1 || this->floatingImage->nt > 1)
        NR_FATAL_ERROR("Multiple time points are not yet supported");
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
                           const size_t activeVoxelNumber) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexture = Cuda::CreateTextureObject(referenceImageCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto warpedTexture = Cuda::CreateTextureObject(warpedCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, activeVoxelNumber, cudaChannelFormatKindSigned, 1);
    Cuda::UniqueTextureObjectPtr localWeightSimTexture;
    if (localWeightSimCuda)
        localWeightSimTexture = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);

    // Create an array on the device to store the absolute difference values
    thrust::device_vector<float> ssdSum(1), ssdCount(1);

    // Compute the absolute values
    const unsigned blocks = CudaContext::GetBlockSize()->GetSsdValue;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    Cuda::GetSsdValueKernel<<<gridDims, blockDims>>>(ssdSum.data().get(), ssdCount.data().get(), *referenceTexture,
                                                     *warpedTexture, localWeightSimCuda ? *localWeightSimTexture : 0,
                                                     *maskTexture, referenceImageDim, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    // Calculate the SSD
    const float ssd = ssdSum[0] / ssdCount[0];

    return -ssd;
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueFw() {
    return reg_getSsdValue_gpu(this->referenceImage,
                               this->referenceImageCuda,
                               this->warpedImageCuda,
                               this->localWeightSimCuda,
                               this->referenceMaskCuda,
                               this->activeVoxelNumber);
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueBw() {
    return reg_getSsdValue_gpu(this->floatingImage,
                               this->floatingImageCuda,
                               this->warpedImageBwCuda,
                               nullptr,
                               this->floatingMaskCuda,
                               this->activeVoxelNumber);
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
                                      const float timePointWeight) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexturePtr = Cuda::CreateTextureObject(referenceImageCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto warpedTexturePtr = Cuda::CreateTextureObject(warpedCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto maskTexturePtr = Cuda::CreateTextureObject(maskCuda, activeVoxelNumber, cudaChannelFormatKindSigned, 1);
    auto spatialGradTexturePtr = Cuda::CreateTextureObject(spatialGradCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    Cuda::UniqueTextureObjectPtr localWeightSimTexturePtr;
    if (localWeightSimCuda)
        localWeightSimTexturePtr = Cuda::CreateTextureObject(localWeightSimCuda, voxelNumber, cudaChannelFormatKindFloat, 1);

    // Find number of valid voxels and correct weight
    const auto referenceTexture = *referenceTexturePtr;
    const auto warpedTexture = *warpedTexturePtr;
    const size_t validVoxelNumber = thrust::count_if(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int index) {
        const float refValue = tex1Dfetch<float>(referenceTexture, index);
        if (refValue != refValue) return false;
        const float warValue = tex1Dfetch<float>(warpedTexture, index);
        if (warValue != warValue) return false;
        return true;
    });
    const float adjustedWeight = timePointWeight / static_cast<float>(validVoxelNumber);

    const unsigned blocks = CudaContext::GetBlockSize()->GetSsdGradient;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    Cuda::GetSsdGradientKernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexturePtr, *warpedTexturePtr, *maskTexturePtr,
                                                        *spatialGradTexturePtr, localWeightSimCuda ? *localWeightSimTexturePtr : 0,
                                                        referenceImageDim, adjustedWeight, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
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
                                     static_cast<float>(this->timePointWeights[currentTimePoint]));
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
                                     static_cast<float>(this->timePointWeights[currentTimePoint]));
}
/* *************************************************************** */
