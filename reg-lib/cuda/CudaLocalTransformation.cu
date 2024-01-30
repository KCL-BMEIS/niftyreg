/*
 *  CudaLocalTransformation.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "CudaLocalTransformation.hpp"
#include "CudaLocalTransformationKernels.cu"
#include "CudaGlobalTransformation.hpp"
#include "_reg_splineBasis.h"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool composition, bool bspline>
void GetDeformationField(const nifti_image *controlPointImage,
                         const nifti_image *referenceImage,
                         const float4 *controlPointImageCuda,
                         float4 *deformationFieldCuda,
                         const int *maskCuda,
                         const size_t activeVoxelNumber) {
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointVoxelSpacing = make_float3(controlPointImage->dx / referenceImage->dx,
                                                        controlPointImage->dy / referenceImage->dy,
                                                        controlPointImage->dz / referenceImage->dz);

    auto controlPointTexturePtr = Cuda::CreateTextureObject(controlPointImageCuda, controlPointNumber, cudaChannelFormatKindFloat, 4);
    auto controlPointTexture = *controlPointTexturePtr;

    // Get the reference matrix if composition is required
    thrust::device_vector<mat44> realToVoxelCudaVec;
    if constexpr (composition) {
        const mat44 *matPtr = controlPointImage->sform_code > 0 ? &controlPointImage->sto_ijk : &controlPointImage->qto_ijk;
        realToVoxelCudaVec = thrust::device_vector<mat44>(matPtr, matPtr + 1);
    }
    const auto realToVoxelCuda = composition ? realToVoxelCudaVec.data().get() : nullptr;

    if (referenceImage->nz > 1) {
        thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
            GetDeformationField3d<composition, bspline>(deformationFieldCuda, controlPointTexture, realToVoxelCuda,
                                                        referenceImageDim, controlPointImageDim, controlPointVoxelSpacing, index);
        });
    } else {
        thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
            GetDeformationField2d<composition, bspline>(deformationFieldCuda, controlPointTexture, realToVoxelCuda,
                                                        referenceImageDim, controlPointImageDim, controlPointVoxelSpacing, index);
        });
    }
}
template void GetDeformationField<false, false>(const nifti_image*, const nifti_image*, const float4*, float4*, const int*, const size_t);
template void GetDeformationField<true, false>(const nifti_image*, const nifti_image*, const float4*, float4*, const int*, const size_t);
/* *************************************************************** */
template<bool is3d>
struct Basis2nd {
    float xx[27], yy[27], zz[27], xy[27], yz[27], xz[27];
};
template<>
struct Basis2nd<false> {
    float xx[9], yy[9], xy[9];
};
template<bool is3d>
struct SecondDerivative {
    using Type = float3;
    using TextureType = float4; // Due to float3 is not allowed for textures
    Type xx, yy, zz, xy, yz, xz;
};
template<>
struct SecondDerivative<false> {
    using Type = float2;
    using TextureType = float2;
    Type xx, yy, xy;
};
/* *************************************************************** */
template<bool is3d, bool isGradient>
__device__ SecondDerivative<is3d> GetApproxSecondDerivative(const int index,
                                                            cudaTextureObject_t controlPointTexture,
                                                            const int3 controlPointImageDim,
                                                            const Basis2nd<is3d> basis) {
    const auto [x, y, z] = reg_indexToDims_cuda<is3d>(index, controlPointImageDim);
    if (!isGradient && (x < 1 || x >= controlPointImageDim.x - 1 ||
                        y < 1 || y >= controlPointImageDim.y - 1 ||
                        (is3d && (z < 1 || z >= controlPointImageDim.z - 1)))) return {};

    SecondDerivative<is3d> secondDerivative{};
    if constexpr (is3d) {
        for (int c = z - 1, basInd = 0; c < z + 2; c++) {
            if (isGradient && (c < 0 || c >= controlPointImageDim.z)) { basInd += 9; continue; }
            const int indexZ = c * controlPointImageDim.y;
            for (int b = y - 1; b < y + 2; b++) {
                if (isGradient && (b < 0 || b >= controlPointImageDim.y)) { basInd += 3; continue; }
                int indexXYZ = (indexZ + b) * controlPointImageDim.x + x - 1;
                for (int a = x - 1; a < x + 2; a++, basInd++, indexXYZ++) {
                    if (isGradient && (a < 0 || a >= controlPointImageDim.x)) continue;
                    const float3 controlPointValue = make_float3(tex1Dfetch<float4>(controlPointTexture, indexXYZ));
                    secondDerivative.xx = secondDerivative.xx + basis.xx[basInd] * controlPointValue;
                    secondDerivative.yy = secondDerivative.yy + basis.yy[basInd] * controlPointValue;
                    secondDerivative.zz = secondDerivative.zz + basis.zz[basInd] * controlPointValue;
                    secondDerivative.xy = secondDerivative.xy + basis.xy[basInd] * controlPointValue;
                    secondDerivative.yz = secondDerivative.yz + basis.yz[basInd] * controlPointValue;
                    secondDerivative.xz = secondDerivative.xz + basis.xz[basInd] * controlPointValue;
                }
            }
        }
    } else {
        for (int b = y - 1, basInd = 0; b < y + 2; b++) {
            if (isGradient && (b < 0 || b >= controlPointImageDim.y)) { basInd += 3; continue; }
            int indexXY = b * controlPointImageDim.x + x - 1;
            for (int a = x - 1; a < x + 2; a++, basInd++, indexXY++) {
                if (isGradient && (a < 0 || a >= controlPointImageDim.x)) continue;
                const float2 controlPointValue = make_float2(tex1Dfetch<float4>(controlPointTexture, indexXY));
                secondDerivative.xx = secondDerivative.xx + basis.xx[basInd] * controlPointValue;
                secondDerivative.yy = secondDerivative.yy + basis.yy[basInd] * controlPointValue;
                secondDerivative.xy = secondDerivative.xy + basis.xy[basInd] * controlPointValue;
            }
        }
    }
    return secondDerivative;
}
/* *************************************************************** */
template<bool is3d>
double ApproxBendingEnergy(const nifti_image *controlPointImage, const float4 *controlPointImageCuda) {
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    auto controlPointTexturePtr = Cuda::CreateTextureObject(controlPointImageCuda, controlPointNumber, cudaChannelFormatKindFloat, 4);
    auto controlPointTexture = *controlPointTexturePtr;

    // Get the constant basis values
    Basis2nd<is3d> basis;
    if constexpr (is3d)
        set_second_order_bspline_basis_values(basis.xx, basis.yy, basis.zz, basis.xy, basis.yz, basis.xz);
    else
        set_second_order_bspline_basis_values(basis.xx, basis.yy, basis.xy);

    thrust::counting_iterator index(0);
    return thrust::transform_reduce(thrust::device, index, index + controlPointNumber, [=]__device__(const int index) {
        const auto secondDerivative = GetApproxSecondDerivative<is3d, false>(index, controlPointTexture, controlPointImageDim, basis);
        if constexpr (is3d)
            return (Square(secondDerivative.xx.x) + Square(secondDerivative.xx.y) + Square(secondDerivative.xx.z) +
                    Square(secondDerivative.yy.x) + Square(secondDerivative.yy.y) + Square(secondDerivative.yy.z) +
                    Square(secondDerivative.zz.x) + Square(secondDerivative.zz.y) + Square(secondDerivative.zz.z) +
                    2.f * (Square(secondDerivative.xy.x) + Square(secondDerivative.xy.y) + Square(secondDerivative.xy.z) +
                           Square(secondDerivative.yz.x) + Square(secondDerivative.yz.y) + Square(secondDerivative.yz.z) +
                           Square(secondDerivative.xz.x) + Square(secondDerivative.xz.y) + Square(secondDerivative.xz.z)));
        else
            return (Square(secondDerivative.xx.x) + Square(secondDerivative.xx.y) + Square(secondDerivative.yy.x) +
                    Square(secondDerivative.yy.y) + 2.f * (Square(secondDerivative.xy.x) + Square(secondDerivative.xy.y)));
    }, 0.0, thrust::plus<double>()) / static_cast<double>(controlPointImage->nvox);
}
template double ApproxBendingEnergy<false>(const nifti_image*, const float4*);
template double ApproxBendingEnergy<true>(const nifti_image*, const float4*);
/* *************************************************************** */
template<bool is3d>
void ApproxBendingEnergyGradient(nifti_image *controlPointImage,
                                 float4 *controlPointImageCuda,
                                 float4 *transGradientCuda,
                                 float bendingEnergyWeight) {
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    auto controlPointTexturePtr = Cuda::CreateTextureObject(controlPointImageCuda, controlPointNumber, cudaChannelFormatKindFloat, 4);
    auto controlPointTexture = *controlPointTexturePtr;

    // Get the constant basis values
    Basis2nd<is3d> basis;
    if constexpr (is3d)
        set_second_order_bspline_basis_values(basis.xx, basis.yy, basis.zz, basis.xy, basis.yz, basis.xz);
    else
        set_second_order_bspline_basis_values(basis.xx, basis.yy, basis.xy);

    GetDisplacementFromDeformation(controlPointImage, controlPointImageCuda);

    // First compute all the second derivatives
    thrust::device_vector<typename SecondDerivative<is3d>::TextureType> secondDerivativesCudaVec((is3d ? 6 : 3) * controlPointNumber);
    auto secondDerivativesCuda = secondDerivativesCudaVec.data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), controlPointNumber,
                       [controlPointTexture, controlPointImageDim, basis, secondDerivativesCuda]__device__(const int index) {
        const auto secondDerivative = GetApproxSecondDerivative<is3d, true>(index, controlPointTexture, controlPointImageDim, basis);
        if constexpr (is3d) {
            int derInd = 6 * index;
            secondDerivativesCuda[derInd++] = make_float4(secondDerivative.xx);
            secondDerivativesCuda[derInd++] = make_float4(secondDerivative.yy);
            secondDerivativesCuda[derInd++] = make_float4(secondDerivative.zz);
            secondDerivativesCuda[derInd++] = make_float4(2.f * secondDerivative.xy);
            secondDerivativesCuda[derInd++] = make_float4(2.f * secondDerivative.yz);
            secondDerivativesCuda[derInd] = make_float4(2.f * secondDerivative.xz);
        } else {
            int derInd = 3 * index;
            secondDerivativesCuda[derInd++] = secondDerivative.xx;
            secondDerivativesCuda[derInd++] = secondDerivative.yy;
            secondDerivativesCuda[derInd] = 2.f * secondDerivative.xy;
        }
    });

    auto secondDerivativesTexturePtr = Cuda::CreateTextureObject(secondDerivativesCuda, secondDerivativesCudaVec.size(), cudaChannelFormatKindFloat,
                                                                 sizeof(typename SecondDerivative<is3d>::TextureType) / sizeof(float));
    auto secondDerivativesTexture = *secondDerivativesTexturePtr;

    // Compute the gradient
    const float approxRatio = bendingEnergyWeight / (float)controlPointNumber;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), controlPointNumber,
                       [controlPointImageDim, basis, secondDerivativesTexture, transGradientCuda, approxRatio]__device__(const int index) {
        const auto [x, y, z] = reg_indexToDims_cuda<is3d>(index, controlPointImageDim);
        typename SecondDerivative<is3d>::Type gradientValue{};
        if constexpr (is3d) {
            for (int c = z - 1, basInd = 0; c < z + 2; c++) {
                if (c < 0 || c >= controlPointImageDim.z) { basInd += 9; continue; }
                const int indexZ = c * controlPointImageDim.y;
                for (int b = y - 1; b < y + 2; b++) {
                    if (b < 0 || b >= controlPointImageDim.y) { basInd += 3; continue; }
                    int indexXYZ = ((indexZ + b) * controlPointImageDim.x + x - 1) * 6;
                    for (int a = x - 1; a < x + 2; a++, basInd++) {
                        if (a < 0 || a >= controlPointImageDim.x) { indexXYZ += 6; continue; }
                        const float3 secondDerivativeXX = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeXX * basis.xx[basInd];
                        const float3 secondDerivativeYY = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeYY * basis.yy[basInd];
                        const float3 secondDerivativeZZ = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeZZ * basis.zz[basInd];
                        const float3 secondDerivativeXY = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeXY * basis.xy[basInd];
                        const float3 secondDerivativeYZ = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeYZ * basis.yz[basInd];
                        const float3 secondDerivativeXZ = make_float3(tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++));
                        gradientValue = gradientValue + secondDerivativeXZ * basis.xz[basInd];
                    }
                }
            }
        } else {
            for (int b = y - 1, basInd = 0; b < y + 2; b++) {
                if (b < 0 || b >= controlPointImageDim.y) { basInd += 3; continue; }
                int indexXY = (b * controlPointImageDim.x + x - 1) * 3;
                for (int a = x - 1; a < x + 2; a++, basInd++) {
                    if (a < 0 || a >= controlPointImageDim.x) { indexXY += 3; continue; }
                    const float2 secondDerivativeXX = tex1Dfetch<float2>(secondDerivativesTexture, indexXY++);
                    gradientValue = gradientValue + secondDerivativeXX * basis.xx[basInd];
                    const float2 secondDerivativeYY = tex1Dfetch<float2>(secondDerivativesTexture, indexXY++);
                    gradientValue = gradientValue + secondDerivativeYY * basis.yy[basInd];
                    const float2 secondDerivativeXY = tex1Dfetch<float2>(secondDerivativesTexture, indexXY++);
                    gradientValue = gradientValue + secondDerivativeXY * basis.xy[basInd];
                }
            }
        }
        float4 nodeGradVal = transGradientCuda[index];
        nodeGradVal.x += approxRatio * gradientValue.x;
        nodeGradVal.y += approxRatio * gradientValue.y;
        if constexpr (is3d)
            nodeGradVal.z += approxRatio * gradientValue.z;
        transGradientCuda[index] = nodeGradVal;
    });

    GetDeformationFromDisplacement(controlPointImage, controlPointImageCuda);
}
template void ApproxBendingEnergyGradient<false>(nifti_image*, float4*, float4*, float);
template void ApproxBendingEnergyGradient<true>(nifti_image*, float4*, float4*, float);
/* *************************************************************** */
void ComputeApproxJacobianValues(const nifti_image *controlPointImage,
                                 const float4 *controlPointImageCuda,
                                 float *jacobianMatricesCuda,
                                 float *jacobianDetCuda) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, controlPointNumber, cudaChannelFormatKindFloat, 4);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_xyz : &controlPointImage->qto_xyz);

    // The Jacobian matrix is computed for every control point
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->GetApproxJacobianValues3d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        GetApproxJacobianValues3d<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                           controlPointImageDim, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->GetApproxJacobianValues2d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        GetApproxJacobianValues2d<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                           controlPointImageDim, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void ComputeJacobianValues(const nifti_image *controlPointImage,
                           const nifti_image *referenceImage,
                           const float4 *controlPointImageCuda,
                           float *jacobianMatricesCuda,
                           float *jacobianDetCuda) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, controlPointNumber, cudaChannelFormatKindFloat, 4);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_xyz : &controlPointImage->qto_xyz);

    // The Jacobian matrix is computed for every voxel
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->GetJacobianValues3d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        // 8 floats of shared memory are allocated per thread
        const unsigned sharedMemSize = blocks * 8 * sizeof(float);
        GetJacobianValues3d<<<gridDims, blockDims, sharedMemSize>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                                    controlPointImageDim, controlPointSpacing, referenceImageDim,
                                                                    (unsigned)voxelNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->GetJacobianValues2d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        GetJacobianValues2d<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                     controlPointImageDim, controlPointSpacing, referenceImageDim,
                                                     (unsigned)voxelNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
double GetJacobianPenaltyTerm(const nifti_image *referenceImage,
                              const nifti_image *controlPointImage,
                              const float4 *controlPointImageCuda,
                              const bool approx) {
    // The Jacobian matrices and determinants are computed
    float *jacobianMatricesCuda, *jacobianDetCuda;
    size_t jacNumber; double jacSum;
    if (approx) {
        jacNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
        jacSum = (controlPointImage->nx - 2) * (controlPointImage->ny - 2);
        if (controlPointImage->nz > 1)
            jacSum *= controlPointImage->nz - 2;
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        jacSum = static_cast<double>(jacNumber);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatricesCuda));

    // The Jacobian determinant are squared and logged (might not be english but will do)
    const unsigned blocks = CudaContext::GetBlockSize()->LogSquaredValues;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)jacNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    LogSquaredValues<<<gridDims, blockDims>>>(jacobianDetCuda, (unsigned)jacNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    // Perform the reduction
    const double penaltyTermValue = SumReduction(jacobianDetCuda, jacNumber);
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDetCuda));
    return penaltyTermValue / jacSum;
}
/* *************************************************************** */
void GetJacobianPenaltyTermGradient(const nifti_image *referenceImage,
                                    const nifti_image *controlPointImage,
                                    const float4 *controlPointImageCuda,
                                    float4 *transGradientCuda,
                                    const float jacobianWeight,
                                    const bool approx) {
    auto blockSize = CudaContext::GetBlockSize();

    // The Jacobian matrices and determinants are computed
    float *jacobianMatricesCuda, *jacobianDetCuda;
    size_t jacNumber;
    if (approx) {
        jacNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }

    // Need to disorient the Jacobian matrix using the header information - voxel to real conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_ijk : &controlPointImage->qto_ijk);

    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    const float3 weight = make_float3(referenceImage->dx * jacobianWeight / ((float)jacNumber * controlPointImage->dx),
                                      referenceImage->dy * jacobianWeight / ((float)jacNumber * controlPointImage->dy),
                                      referenceImage->dz * jacobianWeight / ((float)jacNumber * controlPointImage->dz));
    auto jacobianDeterminantTexture = Cuda::CreateTextureObject(jacobianDetCuda, jacNumber, cudaChannelFormatKindFloat, 1);
    auto jacobianMatricesTexture = Cuda::CreateTextureObject(jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber,
                                                             cudaChannelFormatKindFloat, 1);
    if (approx) {
        if (controlPointImage->nz > 1) {
            const unsigned blocks = blockSize->ComputeApproxJacGradient3d;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ComputeApproxJacGradient3d<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                                *jacobianMatricesTexture, controlPointImageDim,
                                                                (unsigned)controlPointNumber, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        } else {
            const unsigned blocks = blockSize->ComputeApproxJacGradient2d;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ComputeApproxJacGradient2d<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                                *jacobianMatricesTexture, controlPointImageDim,
                                                                (unsigned)controlPointNumber, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        }
    } else {
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(controlPointImage->dx / referenceImage->dx,
                                                            controlPointImage->dy / referenceImage->dy,
                                                            controlPointImage->dz / referenceImage->dz);
        if (controlPointImage->nz > 1) {
            const unsigned blocks = blockSize->ComputeJacGradient3d;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ComputeJacGradient3d<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                          *jacobianMatricesTexture, controlPointImageDim,
                                                          controlPointVoxelSpacing, (unsigned)controlPointNumber,
                                                          referenceImageDim, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        } else {
            const unsigned blocks = blockSize->ComputeJacGradient2d;
            const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            ComputeJacGradient2d<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                          *jacobianMatricesTexture, controlPointImageDim,
                                                          controlPointVoxelSpacing, (unsigned)controlPointNumber,
                                                          referenceImageDim, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        }
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDetCuda));
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatricesCuda));
}
/* *************************************************************** */
double CorrectFolding(const nifti_image *referenceImage,
                      const nifti_image *controlPointImage,
                      float4 *controlPointImageCuda,
                      const bool approx) {
    auto blockSize = CudaContext::GetBlockSize();

    // The Jacobian matrices and determinants are computed
    float *jacobianMatricesCuda, *jacobianDetCuda;
    size_t jacobianDetSize, jacNumber;
    double jacSum;
    if (approx) {
        jacNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
        jacSum = (controlPointImage->nx - 2) * (controlPointImage->ny - 2) * (controlPointImage->nz - 2);
        jacobianDetSize = jacNumber * sizeof(float);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, 9 * jacobianDetSize));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacobianDetSize));
        ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        jacSum = static_cast<double>(jacNumber);
        jacobianDetSize = jacNumber * sizeof(float);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, 9 * jacobianDetSize));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacobianDetSize));
        ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }

    // Check if the Jacobian determinant average
    float *jacobianDet2Cuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet2Cuda, jacobianDetSize));
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet2Cuda, jacobianDetCuda, jacobianDetSize, cudaMemcpyDeviceToDevice));
    const unsigned blocks = blockSize->LogSquaredValues;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)jacNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    LogSquaredValues<<<gridDims, blockDims>>>(jacobianDet2Cuda, (unsigned)jacNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    float *jacobianDet;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet, jacobianDetSize));
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet, jacobianDet2Cuda, jacobianDetSize, cudaMemcpyDeviceToHost));
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet2Cuda));
    double penaltyTermValue = 0;
    for (int i = 0; i < jacNumber; ++i) penaltyTermValue += jacobianDet[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet));
    penaltyTermValue /= jacSum;
    if (penaltyTermValue == penaltyTermValue) {
        NR_CUDA_SAFE_CALL(cudaFree(jacobianDetCuda));
        NR_CUDA_SAFE_CALL(cudaFree(jacobianMatricesCuda));
        return penaltyTermValue;
    }

    // Need to disorient the Jacobian matrix using the header information - voxel to real conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_ijk : &controlPointImage->qto_ijk);

    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    auto jacobianDeterminantTexture = Cuda::CreateTextureObject(jacobianDetCuda, jacNumber, cudaChannelFormatKindFloat, 1);
    auto jacobianMatricesTexture = Cuda::CreateTextureObject(jacobianMatricesCuda, 9 * jacNumber, cudaChannelFormatKindFloat, 1);
    if (approx) {
        const unsigned blocks = blockSize->ApproxCorrectFolding3d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        ApproxCorrectFolding3d<<<gridDims, blockDims>>>(controlPointImageCuda, *jacobianDeterminantTexture,
                                                        *jacobianMatricesTexture, controlPointImageDim,
                                                        controlPointSpacing, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(controlPointImage->dx / referenceImage->dx,
                                                            controlPointImage->dy / referenceImage->dy,
                                                            controlPointImage->dz / referenceImage->dz);
        const unsigned blocks = blockSize->CorrectFolding3d;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        CorrectFolding3d<<<gridDims, blockDims>>>(controlPointImageCuda, *jacobianDeterminantTexture,
                                                  *jacobianMatricesTexture, controlPointImageDim, controlPointSpacing,
                                                  controlPointVoxelSpacing, (unsigned)controlPointNumber,
                                                  referenceImageDim, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDetCuda));
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatricesCuda));
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<bool is3d, bool reverse = false>
void GetDeformationFromDisplacement(nifti_image *image, float4 *imageCuda) {
    // Bind the qform or sform
    const mat44& affineMatrix = image->sform_code > 0 ? image->sto_xyz : image->qto_xyz;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDim{ image->nx, image->ny, image->nz };

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [=]__device__(const int index) {
        const auto [x, y, z] = reg_indexToDims_cuda<is3d>(index, imageDim);

        const float4 initialPosition{
            float(x) * affineMatrix.m[0][0] + float(y) * affineMatrix.m[0][1] + (is3d ? float(z) * affineMatrix.m[0][2] : 0.f) + affineMatrix.m[0][3],
            float(x) * affineMatrix.m[1][0] + float(y) * affineMatrix.m[1][1] + (is3d ? float(z) * affineMatrix.m[1][2] : 0.f) + affineMatrix.m[1][3],
            is3d ? float(x) * affineMatrix.m[2][0] + float(y) * affineMatrix.m[2][1] + float(z) * affineMatrix.m[2][2] + affineMatrix.m[2][3] : 0.f,
            0.f
        };

        // If reverse, gets displacement from deformation
        imageCuda[index] = reverse ? imageCuda[index] - initialPosition : imageCuda[index] + initialPosition;
    });

    image->intent_code = NIFTI_INTENT_VECTOR;
    memset(image->intent_name, 0, 16);
    strcpy(image->intent_name, "NREG_TRANS");
    if constexpr (reverse) {
        if (image->intent_p1 == DEF_FIELD)
            image->intent_p1 = DISP_FIELD;
        else if (image->intent_p1 == DEF_VEL_FIELD)
            image->intent_p1 = DISP_VEL_FIELD;
    } else {
        if (image->intent_p1 == DISP_FIELD)
            image->intent_p1 = DEF_FIELD;
        else if (image->intent_p1 == DISP_VEL_FIELD)
            image->intent_p1 = DEF_VEL_FIELD;
    }
}
/* *************************************************************** */
void GetDeformationFromDisplacement(nifti_image *image, float4 *imageCuda) {
    if (image->nu == 2)
        GetDeformationFromDisplacement<false>(image, imageCuda);
    else if (image->nu == 3)
        GetDeformationFromDisplacement<true>(image, imageCuda);
    else NR_FATAL_ERROR("Only implemented for 2D or 3D deformation fields");
}
/* *************************************************************** */
void GetDisplacementFromDeformation(nifti_image *image, float4 *imageCuda) {
    if (image->nu == 2)
        GetDeformationFromDisplacement<false, true>(image, imageCuda);
    else if (image->nu == 3)
        GetDeformationFromDisplacement<true, true>(image, imageCuda);
    else NR_FATAL_ERROR("Only implemented for 2D or 3D deformation fields");
}
/* *************************************************************** */
void GetFlowFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                  nifti_image *flowField,
                                  float4 *velocityFieldGridCuda,
                                  float4 *flowFieldCuda,
                                  const int *maskCuda,
                                  const size_t activeVoxelNumber) {
    // Check first if the velocity field is actually a velocity field
    if (velocityFieldGrid->intent_p1 != SPLINE_VEL_GRID)
        NR_FATAL_ERROR("The provided grid is not a velocity field");

    // Initialise the flow field with an identity transformation
    flowField->intent_p1 = DISP_VEL_FIELD;
    GetDeformationFromDisplacement(flowField, flowFieldCuda);

    // fake the number of extension here to avoid the second half of the affine
    const auto oldNumExt = velocityFieldGrid->num_ext;
    if (oldNumExt > 1)
        velocityFieldGrid->num_ext = 1;

    // Copy over the number of required squaring steps
    flowField->intent_p2 = velocityFieldGrid->intent_p2;
    // The initial flow field is generated using cubic B-Spline interpolation/approximation
    GetDeformationField<true, true>(velocityFieldGrid,
                                    flowField,
                                    velocityFieldGridCuda,
                                    flowFieldCuda,
                                    maskCuda,
                                    activeVoxelNumber);

    velocityFieldGrid->num_ext = oldNumExt;
}
/* *************************************************************** */
template<bool is3d>
void DefFieldCompose(const nifti_image *deformationField,
                     const float4 *deformationFieldCuda,
                     float4 *deformationFieldOutCuda) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const int3 referenceImageDims{ deformationField->nx, deformationField->ny, deformationField->nz };
    const mat44& affineMatrixB = deformationField->sform_code > 0 ? deformationField->sto_ijk : deformationField->qto_ijk;
    const mat44& affineMatrixC = deformationField->sform_code > 0 ? deformationField->sto_xyz : deformationField->qto_xyz;
    auto deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto deformationFieldTexture = *deformationFieldTexturePtr;

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [=]__device__(const int index) {
        DefFieldComposeKernel<is3d>(deformationFieldOutCuda, deformationFieldTexture, referenceImageDims, affineMatrixB, affineMatrixC, index);
    });
}
/* *************************************************************** */
void GetDeformationFieldFromFlowField(nifti_image *flowField,
                                      nifti_image *deformationField,
                                      float4 *flowFieldCuda,
                                      float4 *deformationFieldCuda,
                                      const bool updateStepNumber) {
    // Check first if the velocity field is actually a velocity field
    if (flowField->intent_p1 != DEF_VEL_FIELD)
        NR_FATAL_ERROR("The provided field is not a velocity field");

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);

    // Remove the affine component from the flow field
    NiftiImage affineOnly;
    thrust::device_vector<float4> affineOnlyCudaVec;
    if (flowField->num_ext > 0) {
        if (flowField->ext_list[0].edata != nullptr) {
            // Create a field that contains the affine component only
            affineOnly = NiftiImage(deformationField, NiftiImage::Copy::ImageInfo);
            affineOnlyCudaVec.resize(voxelNumber);
            Cuda::GetAffineDeformationField(reinterpret_cast<mat44*>(flowField->ext_list[0].edata),
                                            affineOnly, affineOnlyCudaVec.data().get());
            SubtractImages(flowField, flowFieldCuda, affineOnlyCudaVec.data().get());
        }
    } else GetDisplacementFromDeformation(flowField, flowFieldCuda);

    // Compute the number of scaling value to ensure unfolded transformation
    int squaringNumber = 1;
    if (updateStepNumber || flowField->intent_p2 == 0) {
        // Check the largest value
        float extrema = fabsf(GetMinValue(flowField, flowFieldCuda, -1));
        const float temp = GetMaxValue(flowField, flowFieldCuda, -1);
        extrema = std::max(extrema, temp);
        // Check the values for scaling purpose
        float maxLength;
        if (deformationField->nz > 1)
            maxLength = 0.28f;  // sqrt(0.5^2/3)
        else maxLength = 0.35f; // sqrt(0.5^2/2)
        while (extrema / pow(2.0f, squaringNumber) >= maxLength)
            squaringNumber++;
        // The minimal number of step is set to 6 by default
        squaringNumber = squaringNumber < 6 ? 6 : squaringNumber;
        // Set the number of squaring step in the flow field
        if (fabs(flowField->intent_p2) != squaringNumber)
            NR_WARN("Changing from " << Round(fabs(flowField->intent_p2)) << " to " << abs(squaringNumber) <<
                    " squaring step (equivalent to scaling down by " << (int)pow(2.0f, squaringNumber) << ")");
        // Update the number of squaring step required
        flowField->intent_p2 = static_cast<float>(flowField->intent_p2 >= 0 ? squaringNumber : -squaringNumber);
    } else squaringNumber = static_cast<int>(fabsf(flowField->intent_p2));

    // The displacement field is scaled
    const float scalingValue = 1.f / pow(2.f, static_cast<float>(std::abs(squaringNumber)));
    // Backward/forward deformation field is scaled down
    MultiplyValue(voxelNumber, flowFieldCuda, flowField->intent_p2 < 0 ? -scalingValue : scalingValue);

    // Conversion from displacement to deformation
    GetDeformationFromDisplacement(flowField, flowFieldCuda);

    // The computed scaled deformation field is copied over
    thrust::copy(thrust::device, flowFieldCuda, flowFieldCuda + voxelNumber, deformationFieldCuda);

    // The deformation field is squared
    auto defFieldCompose = deformationField->nz > 1 ? DefFieldCompose<true> : DefFieldCompose<false>;
    for (int i = 0; i < squaringNumber; ++i) {
        // The deformation field is applied to itself
        defFieldCompose(deformationField, deformationFieldCuda, flowFieldCuda);
        // The computed scaled deformation field is copied over
        thrust::copy_n(thrust::device, flowFieldCuda, voxelNumber, deformationFieldCuda);
        NR_DEBUG("Squaring (composition) step " << i + 1 << "/" << squaringNumber);
    }
    // The affine component of the transformation is restored
    if (!affineOnlyCudaVec.empty()) {
        GetDisplacementFromDeformation(deformationField, deformationFieldCuda);
        AddImages(deformationField, deformationFieldCuda, affineOnlyCudaVec.data().get());
    }
    deformationField->intent_p1 = DEF_FIELD;
    deformationField->intent_p2 = 0;
    // If required an affine component is composed
    if (flowField->num_ext > 1)
        Cuda::GetAffineDeformationField<true>(reinterpret_cast<mat44*>(flowField->ext_list[1].edata),
                                              deformationField, deformationFieldCuda);
}
/* *************************************************************** */
void GetDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                 nifti_image *deformationField,
                                 float4 *velocityFieldGridCuda,
                                 float4 *deformationFieldCuda,
                                 const bool updateStepNumber) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);

    // Create a mask array where no voxel is excluded
    thrust::device_vector<int> maskCudaVec(voxelNumber);
    thrust::sequence(maskCudaVec.begin(), maskCudaVec.end());

    // Clean any extension in the deformation field as it is unexpected
    nifti_free_extensions(deformationField);

    // Check if the velocity field is actually a velocity field
    if (velocityFieldGrid->intent_p1 == CUB_SPLINE_GRID) {
        // Use the spline approximation to generate the deformation field
        GetDeformationField<false, true>(velocityFieldGrid,
                                         deformationField,
                                         velocityFieldGridCuda,
                                         deformationFieldCuda,
                                         maskCudaVec.data().get(),
                                         voxelNumber);
    } else if (velocityFieldGrid->intent_p1 == SPLINE_VEL_GRID) {
        // Create an image to store the flow field
        NiftiImage flowField(deformationField, NiftiImage::Copy::ImageInfo);
        flowField.setIntentName("NREG_TRANS"s);
        flowField->intent_code = NIFTI_INTENT_VECTOR;
        flowField->intent_p1 = DEF_VEL_FIELD;
        flowField->intent_p2 = velocityFieldGrid->intent_p2;
        if (velocityFieldGrid->num_ext > 0)
            nifti_copy_extensions(flowField, velocityFieldGrid);

        // Allocate CUDA memory for the flow field
        thrust::device_vector<float4> flowFieldCudaVec(voxelNumber);

        // Generate the velocity field
        GetFlowFieldFromVelocityGrid(velocityFieldGrid, flowField, velocityFieldGridCuda,
                                     flowFieldCudaVec.data().get(), maskCudaVec.data().get(), voxelNumber);
        // Exponentiate the flow field
        GetDeformationFieldFromFlowField(flowField, deformationField, flowFieldCudaVec.data().get(),
                                         deformationFieldCuda, updateStepNumber);
        // Update the number of step required. No action otherwise
        velocityFieldGrid->intent_p2 = flowField->intent_p2;
    } else NR_FATAL_ERROR("The provided input image is not a spline parametrised transformation");
}
/* *************************************************************** */
void GetIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                        float4 *velocityFieldGridCuda,
                                        vector<NiftiImage>& deformationFields,
                                        vector<thrust::device_vector<float4>>& deformationFieldCudaVecs) {
    if (velocityFieldGrid->intent_p1 != SPLINE_VEL_GRID)
        NR_FATAL_ERROR("The provided input image is not a spline parametrised transformation");

    // Create a mask array where no voxel is excluded
    const size_t voxelNumber = deformationFields[0].nVoxelsPerVolume();
    thrust::device_vector<int> maskCudaVec(voxelNumber);
    thrust::sequence(maskCudaVec.begin(), maskCudaVec.end());

    // Create an image to store the flow field
    NiftiImage flowField(deformationFields[0], NiftiImage::Copy::ImageInfo);
    flowField.setIntentName("NREG_TRANS"s);
    flowField->intent_code = NIFTI_INTENT_VECTOR;
    flowField->intent_p1 = DEF_VEL_FIELD;
    flowField->intent_p2 = velocityFieldGrid->intent_p2;
    if (velocityFieldGrid->num_ext > 0)
        nifti_copy_extensions(flowField, velocityFieldGrid);

    // Allocate CUDA memory for the flow field
    thrust::device_vector<float4> flowFieldCudaVec(voxelNumber);
    auto flowFieldCuda = flowFieldCudaVec.data().get();

    // Generate the velocity field
    GetFlowFieldFromVelocityGrid(velocityFieldGrid, flowField, velocityFieldGridCuda,
                                 flowFieldCuda, maskCudaVec.data().get(), voxelNumber);

    // Remove the affine component from the flow field
    NiftiImage affineOnly;
    thrust::device_vector<float4> affineOnlyCudaVec;
    if (flowField->num_ext > 0) {
        if (flowField->ext_list[0].edata != nullptr) {
            // Create a field that contains the affine component only
            affineOnly = NiftiImage(deformationFields[0], NiftiImage::Copy::ImageInfo);
            affineOnlyCudaVec.resize(voxelNumber);
            Cuda::GetAffineDeformationField(reinterpret_cast<mat44*>(flowField->ext_list[0].edata),
                                            affineOnly, affineOnlyCudaVec.data().get());
            SubtractImages(flowField, flowFieldCuda, affineOnlyCudaVec.data().get());
        }
    } else GetDisplacementFromDeformation(flowField, flowFieldCuda);

    // Get the number of scaling value
    int squaringNumber = std::abs(static_cast<int>(velocityFieldGrid->intent_p2));

    // The displacement field is scaled
    const float scalingValue = 1.f / pow(2.f, static_cast<float>(squaringNumber));
    // Backward/forward deformation field is scaled down
    MultiplyValue(voxelNumber, flowFieldCuda, deformationFieldCudaVecs[0].data().get(),
                  flowField->intent_p2 < 0 ? -scalingValue : scalingValue);

    // Conversion from displacement to deformation
    GetDeformationFromDisplacement(deformationFields[0], deformationFieldCudaVecs[0].data().get());

    // The deformation field is squared
    auto defFieldCompose = deformationFields[0]->nz > 1 ? DefFieldCompose<true> : DefFieldCompose<false>;
    for (int i = 0; i < squaringNumber; i++) {
        // The computed scaled deformation field is copied over
        thrust::copy_n(thrust::device, deformationFieldCudaVecs[i].data().get(), voxelNumber, deformationFieldCudaVecs[i + 1].data().get());
        // The deformation field is applied to itself
        defFieldCompose(deformationFields[i], deformationFieldCudaVecs[i].data().get(), deformationFieldCudaVecs[i + 1].data().get());
        NR_DEBUG("Squaring (composition) step " << i + 1 << "/" << squaringNumber);
    }

    // The affine component of the transformation is restored
    if (!affineOnlyCudaVec.empty()) {
        for (int i = 0; i <= squaringNumber; i++) {
            GetDisplacementFromDeformation(deformationFields[i], deformationFieldCudaVecs[i].data().get());
            AddImages(deformationFields[i], deformationFieldCudaVecs[i].data().get(), affineOnlyCudaVec.data().get());
            deformationFields[i]->intent_p1 = DEF_FIELD;
            deformationFields[i]->intent_p2 = 0;
        }
    }
    // If required an affine component is composed
    if (velocityFieldGrid->num_ext > 1) {
        for (int i = 0; i <= squaringNumber; i++)
            Cuda::GetAffineDeformationField<true>(reinterpret_cast<mat44*>(velocityFieldGrid->ext_list[1].edata),
                                                  deformationFields[i], deformationFieldCudaVecs[i].data().get());
    }
}
/* *************************************************************** */
void GetJacobianMatrix(const nifti_image *deformationField,
                       const float4 *deformationFieldCuda,
                       float *jacobianMatricesCuda) {
    const int3 referenceImageDim = make_int3(deformationField->nx, deformationField->ny, deformationField->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const mat33 reorientation = reg_mat44_to_mat33(deformationField->sform_code > 0 ? &deformationField->sto_xyz : &deformationField->qto_xyz);
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, voxelNumber, cudaChannelFormatKindFloat, 4);

    const unsigned blocks = CudaContext::GetBlockSize()->GetJacobianMatrix;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    GetJacobianMatrix3d<<<gridDims, blockDims>>>(jacobianMatricesCuda, *deformationFieldTexture, referenceImageDim,
                                                 (unsigned)voxelNumber, reorientation);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
template<bool is3d>
double ApproxLinearEnergy(const nifti_image *controlPointGrid,
                          const float4 *controlPointGridCuda) {
    const int3 cppDims = make_int3(controlPointGrid->nx, controlPointGrid->ny, controlPointGrid->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(controlPointGrid, 3);

    // Matrix to use to convert the gradient from mm to voxel
    const mat33 reorientation = reg_mat44_to_mat33(controlPointGrid->sform_code > 0 ? &controlPointGrid->sto_ijk : &controlPointGrid->qto_ijk);

    // Store the basis values since they are constant as the value is approximated at the control point positions only
    Basis1st<is3d> basis;
    if constexpr (is3d)
        set_first_order_basis_values(basis.x, basis.y, basis.z);
    else
        set_first_order_basis_values(basis.x, basis.y);

    // Create the control point texture
    auto controlPointTexturePtr = Cuda::CreateTextureObject(controlPointGridCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto controlPointTexture = *controlPointTexturePtr;

    constexpr int matSize = is3d ? 3 : 2;
    thrust::counting_iterator index(0);
    return thrust::transform_reduce(thrust::device, index, index + voxelNumber, [=]__device__(const int index) {
        const mat33 matrix = CreateDisplacementMatrix<is3d>(index, controlPointTexture, cppDims, basis, reorientation);
        double currentValue = 0;
        for (int b = 0; b < matSize; b++)
            for (int a = 0; a < matSize; a++)
                currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a]));
        return currentValue;
    }, 0.0, thrust::plus<double>()) / static_cast<double>(controlPointGrid->nvox);
}
template double ApproxLinearEnergy<false>(const nifti_image*, const float4*);
template double ApproxLinearEnergy<true>(const nifti_image*, const float4*);
/* *************************************************************** */
template<bool is3d>
void ApproxLinearEnergyGradient(const nifti_image *controlPointGrid,
                                const float4 *controlPointGridCuda,
                                float4 *transGradCuda,
                                const float weight) {
    const int3 cppDims = make_int3(controlPointGrid->nx, controlPointGrid->ny, controlPointGrid->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(controlPointGrid, 3);
    const float approxRatio = weight / static_cast<float>(voxelNumber);

    // Matrix to use to convert the gradient from mm to voxel
    const mat33 reorientation = reg_mat44_to_mat33(controlPointGrid->sform_code > 0 ? &controlPointGrid->sto_ijk : &controlPointGrid->qto_ijk);
    const mat33 invReorientation = nifti_mat33_inverse(reorientation);

    // Store the basis values since they are constant as the value is approximated at the control point positions only
    Basis1st<is3d> basis;
    if constexpr (is3d)
        set_first_order_basis_values(basis.x, basis.y, basis.z);
    else
        set_first_order_basis_values(basis.x, basis.y);

    // Create the variable to store the displacement matrices
    thrust::device_vector<mat33> dispMatricesCudaVec(voxelNumber);
    auto dispMatricesCuda = dispMatricesCudaVec.data().get();

    // Create the textures
    auto controlPointTexturePtr = Cuda::CreateTextureObject(controlPointGridCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto dispMatricesTexturePtr = Cuda::CreateTextureObject(dispMatricesCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto controlPointTexture = *controlPointTexturePtr;
    auto dispMatricesTexture = *dispMatricesTexturePtr;

    // Create the displacement matrices
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [=]__device__(const int index) {
        dispMatricesCuda[index] = CreateDisplacementMatrix<is3d>(index, controlPointTexture, cppDims, basis, reorientation);
    });

    // Compute the gradient
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [
        transGradCuda, dispMatricesTexture, cppDims, approxRatio, basis, invReorientation
    ]__device__(const int index) {
        const auto [x, y, z] = reg_indexToDims_cuda<is3d>(index, cppDims);
        auto gradVal = transGradCuda[index];

        if constexpr (is3d) {
            for (int c = -1, basInd = 0; c < 2; c++) {
                const int zInd = (z + c) * cppDims.y;
                for (int b = -1; b < 2; b++) {
                    const int yInd = (zInd + y + b) * cppDims.x;
                    for (int a = -1; a < 2; a++, basInd++) {
                        const int matInd = (yInd + x + a) * 9;   // Multiply with the item count of mat33
                        const float dispMatrix[3]{ tex1Dfetch<float>(dispMatricesTexture, matInd),       // m[0][0]
                                                   tex1Dfetch<float>(dispMatricesTexture, matInd + 4),   // m[1][1]
                                                   tex1Dfetch<float>(dispMatricesTexture, matInd + 8) }; // m[2][2]
                        const float gradValues[3]{ -2.f * dispMatrix[0] * basis.x[basInd],
                                                   -2.f * dispMatrix[1] * basis.y[basInd],
                                                   -2.f * dispMatrix[2] * basis.z[basInd] };

                        gradVal.x += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                    invReorientation.m[0][1] * gradValues[1] +
                                                    invReorientation.m[0][2] * gradValues[2]);
                        gradVal.y += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                    invReorientation.m[1][1] * gradValues[1] +
                                                    invReorientation.m[1][2] * gradValues[2]);
                        gradVal.z += approxRatio * (invReorientation.m[2][0] * gradValues[0] +
                                                    invReorientation.m[2][1] * gradValues[1] +
                                                    invReorientation.m[2][2] * gradValues[2]);
                    }
                }
            }
        } else {
            for (int b = -1, basInd = 0; b < 2; b++) {
                const int yInd = (y + b) * cppDims.x;
                for (int a = -1; a < 2; a++, basInd++) {
                    const int matInd = (yInd + x + a) * 9;   // Multiply with the item count of mat33
                    const float dispMatrix[2]{ tex1Dfetch<float>(dispMatricesTexture, matInd),       // m[0][0]
                                               tex1Dfetch<float>(dispMatricesTexture, matInd + 4) }; // m[1][1]
                    const float gradValues[2]{ -2.f * dispMatrix[0] * basis.x[basInd],
                                               -2.f * dispMatrix[1] * basis.y[basInd] };

                    gradVal.x += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                invReorientation.m[0][1] * gradValues[1]);
                    gradVal.y += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                invReorientation.m[1][1] * gradValues[1]);
                }
            }
        }

        transGradCuda[index] = gradVal;
    });
}
template void ApproxLinearEnergyGradient<false>(const nifti_image*, const float4*, float4*, const float);
template void ApproxLinearEnergyGradient<true>(const nifti_image*, const float4*, float4*, const float);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
