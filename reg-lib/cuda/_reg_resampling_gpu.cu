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
void reg_resampleImage_gpu(const nifti_image *floatingImage,
                           float *warpedImageCuda,
                           const cudaArray *floatingImageCuda,
                           const float4 *deformationFieldCuda,
                           const int *maskCuda,
                           const size_t& activeVoxelNumber,
                           const float& paddingValue) {
    auto blockSize = CudaContext::GetBlockSize();
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    // Create the texture object for the floating image
    auto floatingTexture = Cuda::CreateTextureObject(floatingImageCuda, cudaResourceTypeArray);
    // Create the texture object for the deformation field
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, cudaResourceTypeLinear,
                                                             activeVoxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);
    // Create the texture object for the mask
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                 cudaChannelFormatKindSigned, 1);

    // Bind the real to voxel matrix to the texture
    const mat44 floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    if (floatingImage->nz > 1) {
        const unsigned blocks = blockSize->reg_resampleImage3D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_resampleImage3D_kernel<<<gridDims, blockDims>>>(warpedImageCuda, *floatingTexture, *deformationFieldTexture, *maskTexture,
                                                            floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_resampleImage2D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_resampleImage2D_kernel<<<gridDims, blockDims>>>(warpedImageCuda, *floatingTexture, *deformationFieldTexture, *maskTexture,
                                                            floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_getImageGradient_gpu(const nifti_image *floatingImage,
                              const cudaArray *floatingImageCuda,
                              const float4 *deformationFieldCuda,
                              float4 *warpedGradientCuda,
                              const size_t& activeVoxelNumber,
                              const float& paddingValue) {
    auto blockSize = CudaContext::GetBlockSize();
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    // Create the texture object for the floating image
    auto floatingTexture = Cuda::CreateTextureObject(floatingImageCuda, cudaResourceTypeArray);
    // Create the texture object for the deformation field
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, cudaResourceTypeLinear,
                                                             activeVoxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    // Bind the real to voxel matrix to the texture
    const mat44 floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    if (floatingImage->nz > 1) {
        const unsigned blocks = blockSize->reg_getImageGradient3D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getImageGradient3D_kernel<<<gridDims, blockDims>>>(warpedGradientCuda, *floatingTexture, *deformationFieldTexture,
                                                               floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_getImageGradient2D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getImageGradient2D_kernel<<<gridDims, blockDims>>>(warpedGradientCuda, *floatingTexture, *deformationFieldTexture,
                                                               floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
