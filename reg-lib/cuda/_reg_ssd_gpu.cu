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
    printf("[NiftyReg DEBUG] reg_ssd_gpu constructor called\n");
#endif
}
/* *************************************************************** */
void reg_ssd_gpu::InitialiseMeasure(nifti_image *refImgPtr,
                                    nifti_image *floImgPtr,
                                    int *maskRefPtr,
                                    size_t activeVoxNum,
                                    nifti_image *warFloImgPtr,
                                    nifti_image *warFloGraPtr,
                                    nifti_image *forVoxBasedGraPtr,
                                    nifti_image *localWeightSimPtr,
                                    cudaArray *refDevicePtr,
                                    cudaArray *floDevicePtr,
                                    int *refMskDevicePtr,
                                    float *warFloDevicePtr,
                                    float4 *warFloGradDevicePtr,
                                    float4 *forVoxBasedGraDevicePtr) {
    reg_ssd::InitialiseMeasure(refImgPtr,
                               floImgPtr,
                               maskRefPtr,
                               warFloImgPtr,
                               warFloGraPtr,
                               forVoxBasedGraPtr,
                               localWeightSimPtr);
    // Check if a symmetric measure is required
    if (this->isSymmetric) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] Symmetric scheme is not yet supported on the GPU\n");
        reg_exit();
    }
    // Check that the input image are of type float
    if (this->referenceImagePointer->datatype != NIFTI_TYPE_FLOAT32 ||
        this->warpedFloatingImagePointer->datatype != NIFTI_TYPE_FLOAT32) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] The input images are expected to be float\n");
        reg_exit();
    }
    // Check that the input images have only one time point
    if (this->referenceImagePointer->nt > 1 || this->floatingImagePointer->nt > 1) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images should have only one time point\n");
        reg_exit();
    }
    // Bind the required pointers
    this->referenceDevicePointer = refDevicePtr;
    this->floatingDevicePointer = floDevicePtr;
    this->referenceMaskDevicePointer = refMskDevicePtr;
    this->activeVoxelNumber = activeVoxNum;
    this->warpedFloatingDevicePointer = warFloDevicePtr;
    this->warpedFloatingGradientDevicePointer = warFloGradDevicePtr;
    this->forwardVoxelBasedGradientDevicePointer = forVoxBasedGraDevicePtr;
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
    const double SSDValue = reg_getSSDValue_gpu(this->referenceImagePointer,
                                                this->referenceDevicePointer,
                                                this->warpedFloatingDevicePointer,
                                                this->referenceMaskDevicePointer,
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
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) {
    reg_getVoxelBasedSSDGradient_gpu(this->referenceImagePointer,
                                     this->referenceDevicePointer,
                                     this->warpedFloatingDevicePointer,
                                     this->warpedFloatingGradientDevicePointer,
                                     this->forwardVoxelBasedGradientDevicePointer,
                                     1.f,
                                     this->referenceMaskDevicePointer,
                                     this->activeVoxelNumber);
}
/* *************************************************************** */
