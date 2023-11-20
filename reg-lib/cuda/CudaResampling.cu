/*
 *  CudaResampling.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "CudaResampling.hpp"
#include "CudaResamplingKernels.cu"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool is3d>
void ResampleImage(const nifti_image *floatingImage,
                   const float *floatingImageCuda,
                   const nifti_image *warpedImage,
                   float *warpedImageCuda,
                   const float4 *deformationFieldCuda,
                   const int *maskCuda,
                   const size_t activeVoxelNumber,
                   const int interpolation,
                   const float paddingValue) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported on the GPU");

    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, activeVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, activeVoxelNumber, cudaChannelFormatKindSigned, 1);
    // Bind the real to voxel matrix to the texture
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    for (int t = 0; t < warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG((is3d ? "3" : "2") << "D resampling of volume number " << t);
        auto floatingTexture = Cuda::CreateTextureObject(floatingImageCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        if constexpr (is3d) {
            const unsigned blocks = blockSize->reg_resampleImage3D;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ResampleImage3D<<<gridDims, blockDims>>>(warpedImageCuda + t * voxelNumber, *floatingTexture, *deformationFieldTexture, *maskTexture,
                                                     floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        } else {
            const unsigned blocks = blockSize->reg_resampleImage2D;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ResampleImage2D<<<gridDims, blockDims>>>(warpedImageCuda + t * voxelNumber, *floatingTexture, *deformationFieldTexture, *maskTexture,
                                                     floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        }
    }
}
template void ResampleImage<false>(const nifti_image*, const float*, const nifti_image*, float*, const float4*, const int*, const size_t, const int, const float);
template void ResampleImage<true>(const nifti_image*, const float*, const nifti_image*, float*, const float4*, const int*, const size_t, const int, const float);
/* *************************************************************** */
void GetImageGradient(const nifti_image *floatingImage,
                      const float *floatingImageCuda,
                      const float4 *deformationFieldCuda,
                      float4 *warpedGradientCuda,
                      const size_t activeVoxelNumber,
                      const int interpolation,
                      float paddingValue,
                      const int activeTimePoint) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported on the GPU");

    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    if (paddingValue != paddingValue) paddingValue = 0;
    auto floatingTexture = Cuda::CreateTextureObject(floatingImageCuda + activeTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, activeVoxelNumber, cudaChannelFormatKindFloat, 4);
    // Bind the real to voxel matrix to the texture
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    if (floatingImage->nz > 1) {
        const unsigned blocks = blockSize->reg_getImageGradient3D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        GetImageGradient3D<<<gridDims, blockDims>>>(warpedGradientCuda, *floatingTexture, *deformationFieldTexture,
                                                    floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_getImageGradient2D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        GetImageGradient2D<<<gridDims, blockDims>>>(warpedGradientCuda, *floatingTexture, *deformationFieldTexture,
                                                    floatingMatrix, floatingDim, (unsigned)activeVoxelNumber, paddingValue);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
