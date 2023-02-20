/*
 *  _reg_resampling_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_resampling_gpu.h"
#include "_reg_resampling_kernels.cu"

/* *************************************************************** */
void reg_resampleImage_gpu(nifti_image *floatingImage,
                           float *warpedImageArray_d,
                           cudaArray *floatingImageArray_d,
                           float4 *deformationFieldImageArray_d,
                           int *mask_d,
                           size_t activeVoxelNumber,
                           float paddingValue) {
    // Get the BlockSize - The values have been set in CudaContextSingleton
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::GetInstance(0);

    int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    // Create texture object for the floating image
    auto&& floatingTexture = cudaCommon_createTextureObject(floatingImageArray_d, cudaResourceTypeArray);

    // Create texture object for the deformation field
    auto&& deformationFieldTexture = cudaCommon_createTextureObject(deformationFieldImageArray_d, cudaResourceTypeLinear,
                                                                    false, activeVoxelNumber * sizeof(float4),
                                                                    cudaChannelFormatKindFloat, 4, cudaFilterModePoint);

    // Create texture object for the mask
    auto&& maskTexture = cudaCommon_createTextureObject(mask_d, cudaResourceTypeLinear, false, activeVoxelNumber * sizeof(int),
                                                        cudaChannelFormatKindSigned, 1, cudaFilterModePoint);

    // Bind the real to voxel matrix to texture
    mat44 floatingMatrix;
    if (floatingImage->sform_code > 0)
        floatingMatrix = floatingImage->sto_ijk;
    else floatingMatrix = floatingImage->qto_ijk;

    if (floatingImage->nz > 1) {
        const unsigned Grid_reg_resamplefloatingImage3D = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)NR_BLOCK->Block_reg_resampleImage3D));
        dim3 B1(NR_BLOCK->Block_reg_resampleImage3D, 1, 1);
        dim3 G1(Grid_reg_resamplefloatingImage3D, Grid_reg_resamplefloatingImage3D, 1);
        reg_resampleImage3D_kernel<<<G1, B1>>>(warpedImageArray_d, *floatingTexture, *deformationFieldTexture, *maskTexture, floatingMatrix, floatingDim, activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(G1, B1);
    } else {
        const unsigned Grid_reg_resamplefloatingImage2D = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)NR_BLOCK->Block_reg_resampleImage2D));
        dim3 B1(NR_BLOCK->Block_reg_resampleImage2D, 1, 1);
        dim3 G1(Grid_reg_resamplefloatingImage2D, Grid_reg_resamplefloatingImage2D, 1);
        reg_resampleImage2D_kernel<<<G1, B1>>>(warpedImageArray_d, *floatingTexture, *deformationFieldTexture, *maskTexture, floatingMatrix, floatingDim, activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(G1, B1);
    }
}
/* *************************************************************** */
void reg_getImageGradient_gpu(nifti_image *floatingImage,
                              cudaArray *floatingImageArray_d,
                              float4 *deformationFieldImageArray_d,
                              float4 *warpedGradientArray_d,
                              size_t activeVoxelNumber,
                              float paddingValue) {
    // Get the BlockSize - The values have been set in CudaContextSingleton
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::GetInstance(0);

    int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    // Create texture object for the floating image
    auto&& floatingTexture = cudaCommon_createTextureObject(floatingImageArray_d, cudaResourceTypeArray, true);

    // Create texture object for the deformation field
    auto&& deformationFieldTexture = cudaCommon_createTextureObject(deformationFieldImageArray_d, cudaResourceTypeLinear,
                                                                    false, activeVoxelNumber * sizeof(float4),
                                                                    cudaChannelFormatKindFloat, 4, cudaFilterModePoint);

    // Bind the real to voxel matrix to texture
    mat44 floatingMatrix;
    if (floatingImage->sform_code > 0)
        floatingMatrix = floatingImage->sto_ijk;
    else floatingMatrix = floatingImage->qto_ijk;

    if (floatingImage->nz > 1) {
        const unsigned Grid_reg_getImageGradient3D = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)NR_BLOCK->Block_reg_getImageGradient3D));
        dim3 B1(NR_BLOCK->Block_reg_getImageGradient3D, 1, 1);
        dim3 G1(Grid_reg_getImageGradient3D, Grid_reg_getImageGradient3D, 1);
        reg_getImageGradient3D_kernel<<<G1, B1>>>(warpedGradientArray_d, *floatingTexture, *deformationFieldTexture, floatingMatrix, floatingDim, activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(G1, B1);
    } else {
        const unsigned Grid_reg_getImageGradient2D = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)NR_BLOCK->Block_reg_getImageGradient2D));
        dim3 B1(NR_BLOCK->Block_reg_getImageGradient2D, 1, 1);
        dim3 G1(Grid_reg_getImageGradient2D, Grid_reg_getImageGradient2D, 1);
        reg_getImageGradient2D_kernel<<<G1, B1>>>(warpedGradientArray_d, *floatingTexture, *deformationFieldTexture, floatingMatrix, floatingDim, activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(G1, B1);
    }
}
/* *************************************************************** */
