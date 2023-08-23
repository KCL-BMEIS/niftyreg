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
#include <thrust/device_vector.h>

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
                                    nifti_image *localWeightSim,
                                    int *floMask, int *floMaskCuda,
                                    nifti_image *warpedImgBw, float *warpedImgBwCuda,
                                    nifti_image *warpedGradBw, float4 *warpedGradBwCuda,
                                    nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda) {
    reg_ssd::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum, warpedImg, warpedImgCuda,
                                       warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda, localWeightSim, floMask, floMaskCuda,
                                       warpedImgBw, warpedImgBwCuda, warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda);
    // Check that the input images have only one time point
    if (this->referenceImage->nt > 1 || this->floatingImage->nt > 1)
        NR_FATAL_ERROR("Multiple timepoints are not yet supported");
    NR_FUNC_CALLED();
}
/* *************************************************************** */
double reg_getSsdValue_gpu(const nifti_image *referenceImage,
                           const cudaArray *referenceImageCuda,
                           const float *warpedCuda,
                           const int *maskCuda,
                           const size_t& activeVoxelNumber) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexture = cudaCommon_createTextureObject(referenceImageCuda, cudaResourceTypeArray, 0,
                                                           cudaChannelFormatKindNone, 1, cudaFilterModePoint, true);
    auto warpedTexture = cudaCommon_createTextureObject(warpedCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                        cudaChannelFormatKindFloat, 1);
    auto maskTexture = cudaCommon_createTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                      cudaChannelFormatKindSigned, 1);

    // Create an array on the device to store the absolute difference values
    thrust::device_vector<float> absoluteValuesCuda(activeVoxelNumber);

    // Compute the absolute values
    const unsigned blocks = NiftyReg::CudaContext::GetBlockSize()->reg_getSquaredDifference;
    const unsigned grids = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    if (referenceImageDim.z > 1)
        reg_getSquaredDifference3d_kernel<<<gridDims, blockDims>>>(absoluteValuesCuda.data().get(), *referenceTexture, *warpedTexture,
                                                                   *maskTexture, referenceImageDim, (unsigned)activeVoxelNumber);
    else reg_getSquaredDifference2d_kernel<<<gridDims, blockDims>>>(absoluteValuesCuda.data().get(), *referenceTexture, *warpedTexture,
                                                                    *maskTexture, referenceImageDim, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    // Perform a reduction on the absolute values
    const double ssd = (double)reg_sumReduction_gpu(absoluteValuesCuda.data().get(), activeVoxelNumber) / (double)activeVoxelNumber;

    return ssd;
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueFw() {
    return -reg_getSsdValue_gpu(this->referenceImage,
                                this->referenceImageCuda,
                                this->warpedImageCuda,
                                this->referenceMaskCuda,
                                this->activeVoxelNumber);
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValueBw() {
    return -reg_getSsdValue_gpu(this->floatingImage,
                                this->floatingImageCuda,
                                this->warpedImageBwCuda,
                                this->floatingMaskCuda,
                                this->activeVoxelNumber);
}
/* *************************************************************** */
void reg_getVoxelBasedSsdGradient_gpu(const nifti_image *referenceImage,
                                      const cudaArray *referenceImageCuda,
                                      const float *warpedCuda,
                                      const float4 *spaGradientCuda,
                                      float4 *ssdGradientCuda,
                                      const float& maxSD,
                                      const int *maskCuda,
                                      const size_t& activeVoxelNumber) {
    // Copy the constant memory variables
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

    auto referenceTexture = cudaCommon_createTextureObject(referenceImageCuda, cudaResourceTypeArray, 0,
                                                           cudaChannelFormatKindNone, 1, cudaFilterModePoint, true);
    auto warpedTexture = cudaCommon_createTextureObject(warpedCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                        cudaChannelFormatKindFloat, 1);
    auto maskTexture = cudaCommon_createTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                      cudaChannelFormatKindSigned, 1);
    auto spaGradientTexture = cudaCommon_createTextureObject(spaGradientCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float4),
                                                             cudaChannelFormatKindFloat, 4);

    // Set the gradient image to zero
    NR_CUDA_SAFE_CALL(cudaMemset(ssdGradientCuda, 0, voxelNumber * sizeof(float4)));

    const unsigned blocks = NiftyReg::CudaContext::GetBlockSize()->reg_getSSDGradient;
    const unsigned grids = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    if (referenceImageDim.z > 1)
        reg_getSsdGradient3d_kernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                             *spaGradientTexture, referenceImageDim, maxSD, (unsigned)activeVoxelNumber);
    else reg_getSsdGradient2d_kernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                              *spaGradientTexture, referenceImageDim, maxSD, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimepoint) {
    reg_getVoxelBasedSsdGradient_gpu(this->referenceImage,
                                     this->referenceImageCuda,
                                     this->warpedImageCuda,
                                     this->warpedGradientCuda,
                                     this->voxelBasedGradientCuda,
                                     1.f,
                                     this->referenceMaskCuda,
                                     this->activeVoxelNumber);
}
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimepoint) {
    reg_getVoxelBasedSsdGradient_gpu(this->floatingImage,
                                     this->floatingImageCuda,
                                     this->warpedImageBwCuda,
                                     this->warpedGradientBwCuda,
                                     this->voxelBasedGradientBwCuda,
                                     1.f,
                                     this->floatingMaskCuda,
                                     this->activeVoxelNumber);
}
/* *************************************************************** */
