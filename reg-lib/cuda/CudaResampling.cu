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
#include "_reg_common_cuda_kernels.cu"

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
                   const nifti_image *deformationField,
                   const float4 *deformationFieldCuda,
                   const int *maskCuda,
                   const size_t activeVoxelNumber,
                   const int interpolation,
                   const float paddingValue) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported on the GPU");

    const size_t floVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const size_t defVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, defVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto deformationFieldTexture = *deformationFieldTexturePtr;
    // Get the real to voxel matrix
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    for (int t = 0; t < warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG((is3d ? "3" : "2") << "D resampling of volume number " << t);
        auto curWarpedCuda = warpedImageCuda + t * floVoxelNumber;
        auto floatingTexturePtr = Cuda::CreateTextureObject(floatingImageCuda + t * floVoxelNumber, floVoxelNumber, cudaChannelFormatKindFloat, 1);
        auto floatingTexture = *floatingTexturePtr;
        thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [
            curWarpedCuda, floatingTexture, deformationFieldTexture, floatingMatrix, floatingDim, paddingValue
        ]__device__(const int index) {
            // Get the real world deformation in the floating space
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

            curWarpedCuda[index] = intensity;
        });
    }
}
template void ResampleImage<false>(const nifti_image*, const float*, const nifti_image*, float*, const nifti_image*, const float4*, const int*, const size_t, const int, const float);
template void ResampleImage<true>(const nifti_image*, const float*, const nifti_image*, float*, const nifti_image*, const float4*, const int*, const size_t, const int, const float);
/* *************************************************************** */
template<bool is3d>
void GetImageGradient(const nifti_image *floatingImage,
                      const float *floatingImageCuda,
                      const float4 *deformationFieldCuda,
                      const nifti_image *warpedGradient,
                      float4 *warpedGradientCuda,
                      const int interpolation,
                      float paddingValue,
                      const int activeTimePoint) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported on the GPU");

    const size_t refVoxelNumber = NiftiImage::calcVoxelNumber(warpedGradient, 3);
    const size_t floVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    if (paddingValue != paddingValue) paddingValue = 0;
    auto floatingTexturePtr = Cuda::CreateTextureObject(floatingImageCuda + activeTimePoint * floVoxelNumber, floVoxelNumber, cudaChannelFormatKindFloat, 1);
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, refVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto floatingTexture = *floatingTexturePtr;
    auto deformationFieldTexture = *deformationFieldTexturePtr;
    // Get the real to voxel matrix
    const mat44& floatingMatrix = floatingImage->sform_code > 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), refVoxelNumber, [
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
template void GetImageGradient<false>(const nifti_image*, const float*, const float4*, const nifti_image*, float4*, const int, float, const int);
template void GetImageGradient<true>(const nifti_image*, const float*, const float4*, const nifti_image*, float4*, const int, float, const int);
/* *************************************************************** */
template<bool is3d>
static float3 GetRealImageSpacing(const nifti_image *image) {
    float3 spacing{};
    float indexVoxel1[3]{}, indexVoxel2[3], realVoxel1[3], realVoxel2[3];
    reg_mat44_mul(&image->sto_xyz, indexVoxel1, realVoxel1);

    indexVoxel2[1] = indexVoxel2[2] = 0; indexVoxel2[0] = 1;
    reg_mat44_mul(&image->sto_xyz, indexVoxel2, realVoxel2);
    spacing.x = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));

    indexVoxel2[0] = indexVoxel2[2] = 0; indexVoxel2[1] = 1;
    reg_mat44_mul(&image->sto_xyz, indexVoxel2, realVoxel2);
    spacing.y = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));

    if constexpr (is3d) {
        indexVoxel2[0] = indexVoxel2[1] = 0; indexVoxel2[2] = 1;
        reg_mat44_mul(&image->sto_xyz, indexVoxel2, realVoxel2);
        spacing.z = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));
    }

    return spacing;
}
/* *************************************************************** */
template<bool is3d> struct Gradient { using Type = float3; };
template<> struct Gradient<false> { using Type = float2; };
/* *************************************************************** */
template<bool is3d>
void ResampleGradient(const nifti_image *floatingImage,
                      const float4 *floatingImageCuda,
                      const nifti_image *warpedImage,
                      float4 *warpedImageCuda,
                      const nifti_image *deformationField,
                      const float4 *deformationFieldCuda,
                      const int *maskCuda,
                      const size_t activeVoxelNumber,
                      const int interpolation,
                      const float paddingValue) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported");

    const size_t floVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const size_t defVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const int3 floatingDims = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
    const int3 defFieldDims = make_int3(deformationField->nx, deformationField->ny, deformationField->nz);
    auto floatingTexturePtr = Cuda::CreateTextureObject(floatingImageCuda, floVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, defVoxelNumber, cudaChannelFormatKindFloat, 4);
    auto floatingTexture = *floatingTexturePtr;
    auto deformationFieldTexture = *deformationFieldTexturePtr;

    // Get the real to voxel matrix
    const mat44& floatingMatrix = floatingImage->sform_code != 0 ? floatingImage->sto_ijk : floatingImage->qto_ijk;

    // The spacing is computed if the sform is defined
    const float3 realSpacing = warpedImage->sform_code > 0 ? GetRealImageSpacing<is3d>(warpedImage) :
                                                             make_float3(warpedImage->dx, warpedImage->dy, warpedImage->dz);

    // Reorientation matrix is assessed in order to remove the rigid component
    const mat33 reorient = nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));

    thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [
        warpedImageCuda, floatingTexture, deformationFieldTexture, floatingMatrix, floatingDims, defFieldDims, realSpacing, reorient, paddingValue
    ]__device__(const int index) {
        // Get the real world deformation in the floating space
        const float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, index);

        // Get the voxel-based deformation in the floating space and compute the linear interpolation
        int3 previous;
        float xBasis[2], yBasis[2], zBasis[2];
        TransformInterpolate<float, is3d>(floatingMatrix, realDeformation, previous, xBasis, yBasis, zBasis);

        typename Gradient<is3d>::Type gradientValue{};
        if constexpr (is3d) {
            for (char c = 0; c < 2; c++) {
                const int z = previous.z + c;
                if (-1 < z && z < floatingDims.z) {
                    for (char b = 0; b < 2; b++) {
                        const int y = previous.y + b;
                        if (-1 < y && y < floatingDims.y) {
                            for (char a = 0; a < 2; a++) {
                                const int x = previous.x + a;
                                const float weight = xBasis[a] * yBasis[b] * zBasis[c];
                                if (-1 < x && x < floatingDims.x) {
                                    const int floIndex = (z * floatingDims.y + y) * floatingDims.x + x;
                                    const float3 intensity = make_float3(tex1Dfetch<float4>(floatingTexture, floIndex));
                                    gradientValue = gradientValue + intensity * weight;
                                } else gradientValue = gradientValue + paddingValue * weight;
                            }
                        } else gradientValue = gradientValue + paddingValue * yBasis[b] * zBasis[c];
                    }
                } else gradientValue = gradientValue + paddingValue * zBasis[c];
            }
        } else {
            for (char b = 0; b < 2; b++) {
                const int y = previous.y + b;
                if (-1 < y && y < floatingDims.y) {
                    for (char a = 0; a < 2; a++) {
                        const int x = previous.x + a;
                        const float weight = xBasis[a] * yBasis[b];
                        if (-1 < x && x < floatingDims.x) {
                            const int floIndex = y * floatingDims.x + x;
                            const float2 intensity = make_float2(tex1Dfetch<float4>(floatingTexture, floIndex));
                            gradientValue = gradientValue + intensity * weight;
                        } else gradientValue = gradientValue + paddingValue * weight;
                    }
                } else gradientValue = gradientValue + paddingValue * yBasis[b];
            }
        }

        // Compute the Jacobian matrix
        constexpr float basis[] = { 1.f, 0.f };
        constexpr float deriv[] = { -1.f, 1.f };
        auto [x, y, z] = reg_indexToDims_cuda<is3d>(index, defFieldDims);
        mat33 jacMat{};
        for (char c = 0; c < (is3d ? 2 : 1); c++) {
            if constexpr (is3d) {
                previous.z = z + c;
                zBasis[0] = basis[c];
                zBasis[1] = deriv[c];
                // Boundary conditions along z - slidding
                if (z == defFieldDims.z - 1) {
                    if (c == 1)
                        previous.z -= 2;
                    zBasis[0] = fabs(zBasis[0] - 1);
                    zBasis[1] *= -1;
                }
            }
            for (char b = 0; b < 2; b++) {
                previous.y = y + b;
                yBasis[0] = basis[b];
                yBasis[1] = deriv[b];
                // Boundary conditions along y - slidding
                if (y == defFieldDims.y - 1) {
                    if (b == 1)
                        previous.y -= 2;
                    yBasis[0] = fabs(yBasis[0] - 1);
                    yBasis[1] *= -1;
                }
                for (char a = 0; a < 2; a++) {
                    previous.x = x + a;
                    xBasis[0] = basis[a];
                    xBasis[1] = deriv[a];
                    // Boundary conditions along x - slidding
                    if (x == defFieldDims.x - 1) {
                        if (a == 1)
                            previous.x -= 2;
                        xBasis[0] = fabs(xBasis[0] - 1);
                        xBasis[1] *= -1;
                    }

                    // Compute the basis function values
                    const float3 weight = make_float3(xBasis[1] * yBasis[0] * (is3d ? zBasis[0] : 1),
                                                      xBasis[0] * yBasis[1] * (is3d ? zBasis[0] : 1),
                                                      is3d ? xBasis[0] * yBasis[0] * zBasis[1] : 0);

                    // Get the deformation field values
                    const int defIndex = ((is3d ? previous.z * defFieldDims.y : 0) + previous.y) * defFieldDims.x + previous.x;
                    const float4 defFieldValue = tex1Dfetch<float4>(deformationFieldTexture, defIndex);

                    // Symmetric difference to compute the derivatives
                    jacMat.m[0][0] += weight.x * defFieldValue.x;
                    jacMat.m[0][1] += weight.y * defFieldValue.x;
                    jacMat.m[1][0] += weight.x * defFieldValue.y;
                    jacMat.m[1][1] += weight.y * defFieldValue.y;
                    if constexpr (is3d) {
                        jacMat.m[0][2] += weight.z * defFieldValue.x;
                        jacMat.m[1][2] += weight.z * defFieldValue.y;
                        jacMat.m[2][0] += weight.x * defFieldValue.z;
                        jacMat.m[2][1] += weight.y * defFieldValue.z;
                        jacMat.m[2][2] += weight.z * defFieldValue.z;
                    }
                }
            }
        }
        // reorient and scale the Jacobian matrix
        jacMat = reg_mat33_mul_cuda(reorient, jacMat);
        jacMat.m[0][0] /= realSpacing.x;
        jacMat.m[0][1] /= realSpacing.y;
        jacMat.m[1][0] /= realSpacing.x;
        jacMat.m[1][1] /= realSpacing.y;
        if constexpr (is3d) {
            jacMat.m[0][2] /= realSpacing.z;
            jacMat.m[1][2] /= realSpacing.z;
            jacMat.m[2][0] /= realSpacing.x;
            jacMat.m[2][1] /= realSpacing.y;
            jacMat.m[2][2] /= realSpacing.z;
        }

        // Modulate the gradient scalar values
        float4 warpedValue{};
        if constexpr (is3d) {
            warpedValue.x = jacMat.m[0][0] * gradientValue.x + jacMat.m[0][1] * gradientValue.y + jacMat.m[0][2] * gradientValue.z;
            warpedValue.y = jacMat.m[1][0] * gradientValue.x + jacMat.m[1][1] * gradientValue.y + jacMat.m[1][2] * gradientValue.z;
            warpedValue.z = jacMat.m[2][0] * gradientValue.x + jacMat.m[2][1] * gradientValue.y + jacMat.m[2][2] * gradientValue.z;
        } else {
            warpedValue.x = jacMat.m[0][0] * gradientValue.x + jacMat.m[0][1] * gradientValue.y;
            warpedValue.y = jacMat.m[1][0] * gradientValue.x + jacMat.m[1][1] * gradientValue.y;
        }
        warpedImageCuda[index] = warpedValue;
    });
}
template void ResampleGradient<false>(const nifti_image*, const float4*, const nifti_image*, float4*, const nifti_image*, const float4*, const int*, const size_t, const int, const float);
template void ResampleGradient<true>(const nifti_image*, const float4*, const nifti_image*, float4*, const nifti_image*, const float4*, const int*, const size_t, const int, const float);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
