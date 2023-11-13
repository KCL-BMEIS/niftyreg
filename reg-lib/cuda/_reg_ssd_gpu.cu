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
void reg_ssd_gpu::InitialiseMeasure(nifti_image *refImg, cudaArray *refImgCuda,
                                    nifti_image *floImg, cudaArray *floImgCuda,
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
            Cuda::TransferNiftiToDevice<float>(this->referenceImageCuda, this->referenceImage);
            Cuda::TransferNiftiToDevice<float>(this->floatingImageCuda, this->floatingImage);
            break;
        }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
double reg_getSsdValue_gpu(const nifti_image *referenceImage,
                           const cudaArray *referenceImageCuda,
                           const float *warpedCuda,
                           const float *localWeightSimCuda,
                           const int *maskCuda,
                           const size_t& activeVoxelNumber) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexture = Cuda::CreateTextureObject(referenceImageCuda, cudaResourceTypeArray);
    auto warpedTexture = Cuda::CreateTextureObject(warpedCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                   cudaChannelFormatKindFloat, 1);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                 cudaChannelFormatKindSigned, 1);
    Cuda::UniqueTextureObjectPtr localWeightSimTexture(nullptr, nullptr);
    if (localWeightSimCuda)
        localWeightSimTexture = std::move(Cuda::CreateTextureObject(localWeightSimCuda, cudaResourceTypeLinear,
                                                                    voxelNumber * sizeof(float), cudaChannelFormatKindFloat, 1));

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
                                      const cudaArray *referenceImageCuda,
                                      const float *warpedCuda,
                                      const float4 *spatialGradCuda,
                                      const float *localWeightSimCuda,
                                      float4 *ssdGradientCuda,
                                      const int *maskCuda,
                                      const size_t& activeVoxelNumber,
                                      const float& timepointWeight) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexture = Cuda::CreateTextureObject(referenceImageCuda, cudaResourceTypeArray);
    auto warpedTexture = Cuda::CreateTextureObject(warpedCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                   cudaChannelFormatKindFloat, 1);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                 cudaChannelFormatKindSigned, 1);
    auto spatialGradTexture = Cuda::CreateTextureObject(spatialGradCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float4),
                                                        cudaChannelFormatKindFloat, 4);
    Cuda::UniqueTextureObjectPtr localWeightSimTexture(nullptr, nullptr);
    if (localWeightSimCuda)
        localWeightSimTexture = std::move(Cuda::CreateTextureObject(localWeightSimCuda, cudaResourceTypeLinear,
                                                                    voxelNumber * sizeof(float), cudaChannelFormatKindFloat, 1));

    // Find number of valid voxels and correct weight
    const cudaTextureObject_t referenceTextureObject = *referenceTexture;
    const cudaTextureObject_t warpedTextureObject = *warpedTexture;
    const size_t validVoxelNumber = thrust::count_if(thrust::device, maskCuda, maskCuda + activeVoxelNumber, [=]__device__(const int& index) {
        const float warValue = tex1Dfetch<float>(warpedTextureObject, index);
        if (warValue != warValue) return false;

        const auto&& [x, y, z] = reg_indexToDims_cuda(index, referenceImageDim);
        const float refValue = tex3D<float>(referenceTextureObject, x, y, z);
        if (refValue != refValue) return false;

        return true;
    });
    const float adjustedWeight = timepointWeight / static_cast<float>(validVoxelNumber);

    const unsigned blocks = CudaContext::GetBlockSize()->GetSsdGradient;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    Cuda::GetSsdGradientKernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                        *spatialGradTexture, localWeightSimCuda ? *localWeightSimTexture : 0,
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
