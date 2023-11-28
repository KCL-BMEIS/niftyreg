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

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<typename T>
__inline__ __device__ void InterpLinearKernel(T relative, T (&basis)[2]) {
    basis[1] = relative;
    basis[0] = 1.f - relative;
}
/* *************************************************************** */
template<typename T, bool is3d>
__inline__ __device__ void TransformInterpolate(const mat44 matrix, const float4 realDeformation, int3& previous,
                                                T (&xBasis)[2], T (&yBasis)[2], T (&zBasis)[2]) {
    // Get the voxel-based deformation
    T voxelDeformation[is3d ? 3 : 2];
    if constexpr (is3d) {
        voxelDeformation[0] = (static_cast<T>(matrix.m[0][0]) * static_cast<T>(realDeformation.x) +
                               static_cast<T>(matrix.m[0][1]) * static_cast<T>(realDeformation.y) +
                               static_cast<T>(matrix.m[0][2]) * static_cast<T>(realDeformation.z) +
                               static_cast<T>(matrix.m[0][3]));
        voxelDeformation[1] = (static_cast<T>(matrix.m[1][0]) * static_cast<T>(realDeformation.x) +
                               static_cast<T>(matrix.m[1][1]) * static_cast<T>(realDeformation.y) +
                               static_cast<T>(matrix.m[1][2]) * static_cast<T>(realDeformation.z) +
                               static_cast<T>(matrix.m[1][3]));
        voxelDeformation[2] = (static_cast<T>(matrix.m[2][0]) * static_cast<T>(realDeformation.x) +
                               static_cast<T>(matrix.m[2][1]) * static_cast<T>(realDeformation.y) +
                               static_cast<T>(matrix.m[2][2]) * static_cast<T>(realDeformation.z) +
                               static_cast<T>(matrix.m[2][3]));
    } else {
        voxelDeformation[0] = (static_cast<T>(matrix.m[0][0]) * static_cast<T>(realDeformation.x) +
                               static_cast<T>(matrix.m[0][1]) * static_cast<T>(realDeformation.y) +
                               static_cast<T>(matrix.m[0][3]));
        voxelDeformation[1] = (static_cast<T>(matrix.m[1][0]) * static_cast<T>(realDeformation.x) +
                               static_cast<T>(matrix.m[1][1]) * static_cast<T>(realDeformation.y) +
                               static_cast<T>(matrix.m[1][3]));
    }

    // Compute the linear interpolation
    previous.x = Floor(voxelDeformation[0]);
    previous.y = Floor(voxelDeformation[1]);
    InterpLinearKernel(voxelDeformation[0] - static_cast<T>(previous.x), xBasis);
    InterpLinearKernel(voxelDeformation[1] - static_cast<T>(previous.y), yBasis);
    if constexpr (is3d) {
        previous.z = Floor(voxelDeformation[2]);
        InterpLinearKernel(voxelDeformation[2] - static_cast<T>(previous.z), zBasis);
    }
}
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

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, activeVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto maskTexturePtr = Cuda::CreateTextureObject(maskCuda, activeVoxelNumber, cudaChannelFormatKindSigned, 1);
    auto deformationFieldTexture = *deformationFieldTexturePtr;
    auto maskTexture = *maskTexturePtr;
    // Bind the real to voxel matrix to the texture
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    for (int t = 0; t < warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG((is3d ? "3" : "2") << "D resampling of volume number " << t);
        auto curWarpedCuda = warpedImageCuda + t * voxelNumber;
        auto floatingTexturePtr = Cuda::CreateTextureObject(floatingImageCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        auto floatingTexture = *floatingTexturePtr;
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), activeVoxelNumber, [
            curWarpedCuda, floatingTexture, deformationFieldTexture, maskTexture, floatingMatrix, floatingDim, paddingValue
        ]__device__(const int index) {
            // Get the real world deformation in the floating space
            const int voxel = tex1Dfetch<int>(maskTexture, index);
            const float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, index);

            // Get the voxel-based deformation in the floating space and compute the linear interpolation
            int3 previous;
            double xBasis[2], yBasis[2], zBasis[2];
            TransformInterpolate<double, is3d>(floatingMatrix, realDeformation, previous, xBasis, yBasis, zBasis);

            double intensity = 0;
            if constexpr (is3d) {
                for (char c = 0; c < 2; c++) {
                    const int z = previous.z + c;
                    int indexYZ = (z * floatingDim.y + previous.y) * floatingDim.x;
                    double tempY = 0;
                    for (char b = 0; b < 2; b++, indexYZ += floatingDim.x) {
                        const int y = previous.y + b;
                        int index = indexYZ + previous.x;
                        double tempX = 0;
                        for (char a = 0; a < 2; a++, index++) {
                            const int x = previous.x + a;
                            if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z) {
                                tempX += tex1Dfetch<float>(floatingTexture, index) * xBasis[a];
                            } else {
                                // Padding value
                                tempX += paddingValue * xBasis[a];
                            }
                        }
                        tempY += tempX * yBasis[b];
                    }
                    intensity += tempY * zBasis[c];
                }
            } else {
                int indexY = previous.y * floatingDim.x + previous.x;
                for (char b = 0; b < 2; b++, indexY += floatingDim.x) {
                    const int y = previous.y + b;
                    int index = indexY;
                    double tempX = 0;
                    for (char a = 0; a < 2; a++, index++) {
                        const int x = previous.x + a;
                        if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y) {
                            tempX += tex1Dfetch<float>(floatingTexture, index) * xBasis[a];
                        } else {
                            // Padding value
                            tempX += paddingValue * xBasis[a];
                        }
                    }
                    intensity += tempX * yBasis[b];
                }
            }

            curWarpedCuda[voxel] = intensity;
        });
    }
}
template void ResampleImage<false>(const nifti_image*, const float*, const nifti_image*, float*, const float4*, const int*, const size_t, const int, const float);
template void ResampleImage<true>(const nifti_image*, const float*, const nifti_image*, float*, const float4*, const int*, const size_t, const int, const float);
/* *************************************************************** */
template<bool is3d>
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

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    if (paddingValue != paddingValue) paddingValue = 0;
    auto floatingTexturePtr = Cuda::CreateTextureObject(floatingImageCuda + activeTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, activeVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto floatingTexture = *floatingTexturePtr;
    auto deformationFieldTexture = *deformationFieldTexturePtr;
    // Bind the real to voxel matrix to the texture
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), activeVoxelNumber, [
        warpedGradientCuda, floatingTexture, deformationFieldTexture, floatingMatrix, floatingDim, paddingValue
    ]__device__(const int index) {
            // Get the real world deformation in the floating space
            float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, index);

            // Get the voxel-based deformation in the floating space and compute the linear interpolation
            int3 previous;
            float xBasis[2], yBasis[2], zBasis[2];
            TransformInterpolate<float, is3d>(floatingMatrix, realDeformation, previous, xBasis, yBasis, zBasis);
            constexpr float deriv[] = { -1.0f, 1.0f };

            float4 gradientValue{};
            if constexpr (is3d) {
                for (char c = 0; c < 2; c++) {
                    const int z = previous.z + c;
                    int indexYZ = (z * floatingDim.y + previous.y) * floatingDim.x;
                    float3 tempY{};
                    for (char b = 0; b < 2; b++, indexYZ += floatingDim.x) {
                        const int y = previous.y + b;
                        int index = indexYZ + previous.x;
                        float2 tempX{};
                        for (char a = 0; a < 2; a++, index++) {
                            const int x = previous.x + a;
                            const float intensity = -1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z ?
                                tex1Dfetch<float>(floatingTexture, index) : paddingValue;

                            tempX.x += intensity * deriv[a];
                            tempX.y += intensity * xBasis[a];
                        }
                        tempY.x += tempX.x * yBasis[b];
                        tempY.y += tempX.y * deriv[b];
                        tempY.z += tempX.y * yBasis[b];
                    }
                    gradientValue.x += tempY.x * zBasis[c];
                    gradientValue.y += tempY.y * zBasis[c];
                    gradientValue.z += tempY.z * deriv[c];
                }
            } else {
                int indexY = previous.y * floatingDim.x + previous.x;
                for (char b = 0; b < 2; b++, indexY += floatingDim.x) {
                    const int y = previous.y + b;
                    int index = indexY;
                    float2 tempX{};
                    for (char a = 0; a < 2; a++, index++) {
                        const int x = previous.x + a;
                        const float intensity = -1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y ?
                            tex1Dfetch<float>(floatingTexture, index) : paddingValue;

                        tempX.x += intensity * deriv[a];
                        tempX.y += intensity * xBasis[a];
                    }
                    gradientValue.x += tempX.x * yBasis[b];
                    gradientValue.y += tempX.y * deriv[b];
                }
            }

            warpedGradientCuda[index] = gradientValue;
    });
}
template void GetImageGradient<false>(const nifti_image*, const float*, const float4*, float4*, const size_t, const int, float, const int);
template void GetImageGradient<true>(const nifti_image*, const float*, const float4*, float4*, const size_t, const int, float, const int);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
