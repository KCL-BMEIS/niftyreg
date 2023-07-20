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
#ifndef NDEBUG
    reg_print_msg_debug("reg_ssd_gpu constructor called");
#endif
}
/* *************************************************************** */
reg_ssd_gpu::~reg_ssd_gpu() {
#ifndef NDEBUG
    reg_print_msg_debug("reg_ssd_gpu destructor called");
#endif
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
    // Check if a symmetric measure is required
    if (this->isSymmetric) {
        reg_print_fct_error("reg_ssd_gpu::InitialiseMeasure");
        reg_print_msg_error("Symmetric scheme is not yet supported");
        reg_exit();
    }
    // Check that the input images have only one time point
    if (this->referenceImage->nt > 1 || this->floatingImage->nt > 1) {
        reg_print_fct_error("reg_ssd_gpu::InitialiseMeasure");
        reg_print_msg_error("Multiple timepoints are not yet supported");
        reg_exit();
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_ssd_gpu::InitialiseMeasure()\n");
#endif
}
/* *************************************************************** */
double reg_getSSDValue_gpu(const nifti_image *referenceImage,
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
    float *absoluteValuesCuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&absoluteValuesCuda, activeVoxelNumber * sizeof(float)));

    // Compute the absolute values
    const unsigned blocks = NiftyReg::CudaContext::GetBlockSize()->reg_getSquaredDifference;
    const unsigned grids = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    if (referenceImageDim.z > 1)
        reg_getSquaredDifference3D_kernel<<<gridDims, blockDims>>>(absoluteValuesCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                                   referenceImageDim, (unsigned)activeVoxelNumber);
    else reg_getSquaredDifference2D_kernel<<<gridDims, blockDims>>>(absoluteValuesCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                                    referenceImageDim, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    // Perform a reduction on the absolute values
    const double ssd = (double)reg_sumReduction_gpu(absoluteValuesCuda, activeVoxelNumber) / (double)activeVoxelNumber;

    // Free the absolute value array
    NR_CUDA_SAFE_CALL(cudaFree(absoluteValuesCuda));

    return ssd;
}
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValue() {
    const double SSDValue = reg_getSSDValue_gpu(this->referenceImage,
                                                this->referenceImageCuda,
                                                this->warpedImageCuda,
                                                this->referenceMaskCuda,
                                                this->activeVoxelNumber);
    return -SSDValue;
}
/* *************************************************************** */
void reg_getVoxelBasedSSDGradient_gpu(const nifti_image *referenceImage,
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
        reg_getSSDGradient3D_kernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                             *spaGradientTexture, referenceImageDim, maxSD, (unsigned)activeVoxelNumber);
    else reg_getSSDGradient2D_kernel<<<gridDims, blockDims>>>(ssdGradientCuda, *referenceTexture, *warpedTexture, *maskTexture,
                                                              *spaGradientTexture, referenceImageDim, maxSD, (unsigned)activeVoxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    reg_getVoxelBasedSSDGradient_gpu(this->referenceImage,
                                     this->referenceImageCuda,
                                     this->warpedImageCuda,
                                     this->warpedGradientCuda,
                                     this->voxelBasedGradientCuda,
                                     1.f,
                                     this->referenceMaskCuda,
                                     this->activeVoxelNumber);
}
/* *************************************************************** */
