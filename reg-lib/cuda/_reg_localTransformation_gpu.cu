/*
 *  _reg_spline_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation_kernels.cu"

/* *************************************************************** */
void reg_spline_getDeformationField_gpu(const nifti_image *controlPointImage,
                                        const nifti_image *referenceImage,
                                        const float4 *controlPointImageCuda,
                                        float4 *deformationFieldCuda,
                                        const int *maskCuda,
                                        const size_t& activeVoxelNumber,
                                        const bool& bspline) {
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointVoxelSpacing = make_float3(controlPointImage->dx / referenceImage->dx,
                                                        controlPointImage->dy / referenceImage->dy,
                                                        controlPointImage->dz / referenceImage->dz);

    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, cudaResourceTypeLinear,
                                                         controlPointNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear,
                                                 activeVoxelNumber * sizeof(int), cudaChannelFormatKindSigned, 1);

    if (referenceImage->nz > 1) {
        const unsigned blocks = CudaContext::GetBlockSize()->reg_spline_getDeformationField3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        // 8 floats of shared memory are allocated per thread
        reg_spline_getDeformationField3D<<<gridDims, blockDims, blocks * 8 * sizeof(float)>>>(deformationFieldCuda,
                                                                                              *controlPointTexture,
                                                                                              *maskTexture,
                                                                                              referenceImageDim,
                                                                                              controlPointImageDim,
                                                                                              controlPointVoxelSpacing,
                                                                                              (unsigned)activeVoxelNumber,
                                                                                              bspline);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = CudaContext::GetBlockSize()->reg_spline_getDeformationField2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        // 4 floats of shared memory are allocated per thread
        reg_spline_getDeformationField2D<<<gridDims, blockDims, blocks * 4 * sizeof(float)>>>(deformationFieldCuda,
                                                                                              *controlPointTexture,
                                                                                              *maskTexture,
                                                                                              referenceImageDim,
                                                                                              controlPointImageDim,
                                                                                              controlPointVoxelSpacing,
                                                                                              (unsigned)activeVoxelNumber,
                                                                                              bspline);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
float reg_spline_approxBendingEnergy_gpu(const nifti_image *controlPointImage, const float4 *controlPointImageCuda) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const size_t controlPointGridSize = controlPointNumber * sizeof(float4);
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, cudaResourceTypeLinear,
                                                         controlPointGridSize, cudaChannelFormatKindFloat, 4);

    // First compute all the second derivatives
    float4 *secondDerivativeValuesCuda;
    size_t secondDerivativeValuesSize;
    if (controlPointImage->nz > 1) {
        secondDerivativeValuesSize = 6 * controlPointGridSize;
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValuesCuda, secondDerivativeValuesSize));
        const unsigned blocks = blockSize->reg_spline_getApproxSecondDerivatives3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxSecondDerivatives3D<<<gridDims, blockDims>>>(secondDerivativeValuesCuda, *controlPointTexture,
                                                                         controlPointImageDim, (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        secondDerivativeValuesSize = 3 * controlPointGridSize;
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValuesCuda, secondDerivativeValuesSize));
        const unsigned blocks = blockSize->reg_spline_getApproxSecondDerivatives2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxSecondDerivatives2D<<<gridDims, blockDims>>>(secondDerivativeValuesCuda, *controlPointTexture,
                                                                         controlPointImageDim, (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }

    // Compute the bending energy from the second derivatives
    float *penaltyTermCuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&penaltyTermCuda, controlPointNumber * sizeof(float)));
    auto secondDerivativesTexture = Cuda::CreateTextureObject(secondDerivativeValuesCuda, cudaResourceTypeLinear,
                                                              secondDerivativeValuesSize, cudaChannelFormatKindFloat, 4);
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->reg_spline_getApproxBendingEnergy3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxBendingEnergy3D_kernel<<<gridDims, blockDims>>>(penaltyTermCuda, *secondDerivativesTexture,
                                                                            (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_spline_getApproxBendingEnergy2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxBendingEnergy2D_kernel<<<gridDims, blockDims>>>(penaltyTermCuda, *secondDerivativesTexture,
                                                                            (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValuesCuda));

    // Compute the mean bending energy value
    double penaltyValue = reg_sumReduction_gpu(penaltyTermCuda, controlPointNumber);
    NR_CUDA_SAFE_CALL(cudaFree(penaltyTermCuda));

    return (float)(penaltyValue / (double)controlPointImage->nvox);
}
/* *************************************************************** */
void reg_spline_approxBendingEnergyGradient_gpu(const nifti_image *controlPointImage,
                                                const float4 *controlPointImageCuda,
                                                float4 *transGradientCuda,
                                                float bendingEnergyWeight) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const size_t controlPointGridSize = controlPointNumber * sizeof(float4);
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, cudaResourceTypeLinear,
                                                         controlPointGridSize, cudaChannelFormatKindFloat, 4);

    // First compute all the second derivatives
    float4 *secondDerivativeValuesCuda;
    size_t secondDerivativeValuesSize;
    if (controlPointImage->nz > 1) {
        secondDerivativeValuesSize = 6 * controlPointGridSize * sizeof(float4);
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValuesCuda, secondDerivativeValuesSize));
        const unsigned blocks = blockSize->reg_spline_getApproxSecondDerivatives3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxSecondDerivatives3D<<<gridDims, blockDims>>>(secondDerivativeValuesCuda, *controlPointTexture,
                                                                         controlPointImageDim, (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        secondDerivativeValuesSize = 3 * controlPointGridSize * sizeof(float4);
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValuesCuda, secondDerivativeValuesSize));
        const unsigned blocks = blockSize->reg_spline_getApproxSecondDerivatives2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxSecondDerivatives2D<<<gridDims, blockDims>>>(secondDerivativeValuesCuda, *controlPointTexture,
                                                                         controlPointImageDim, (unsigned)controlPointNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }

    // Compute the gradient
    bendingEnergyWeight *= 1.f / (float)controlPointNumber;
    auto secondDerivativesTexture = Cuda::CreateTextureObject(secondDerivativeValuesCuda, cudaResourceTypeLinear,
                                                              secondDerivativeValuesSize, cudaChannelFormatKindFloat, 4);
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->reg_spline_getApproxBendingEnergyGradient3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxBendingEnergyGradient3D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *secondDerivativesTexture,
                                                                                    controlPointImageDim, (unsigned)controlPointNumber,
                                                                                    bendingEnergyWeight);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_spline_getApproxBendingEnergyGradient2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxBendingEnergyGradient2D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *secondDerivativesTexture,
                                                                                    controlPointImageDim, (unsigned)controlPointNumber,
                                                                                    bendingEnergyWeight);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValuesCuda));
}
/* *************************************************************** */
void reg_spline_ComputeApproxJacobianValues(const nifti_image *controlPointImage,
                                            const float4 *controlPointImageCuda,
                                            float *jacobianMatricesCuda,
                                            float *jacobianDetCuda) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, cudaResourceTypeLinear,
                                                         controlPointNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_xyz : &controlPointImage->qto_xyz);

    // The Jacobian matrix is computed for every control point
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->reg_spline_getApproxJacobianValues3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxJacobianValues3D_kernel<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                                             controlPointImageDim, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_spline_getApproxJacobianValues2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getApproxJacobianValues2D_kernel<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                                             controlPointImageDim, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_spline_ComputeJacobianValues(const nifti_image *controlPointImage,
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
    auto controlPointTexture = Cuda::CreateTextureObject(controlPointImageCuda, cudaResourceTypeLinear,
                                                         controlPointNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_xyz : &controlPointImage->qto_xyz);

    // The Jacobian matrix is computed for every voxel
    if (controlPointImage->nz > 1) {
        const unsigned blocks = blockSize->reg_spline_getJacobianValues3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        // 8 floats of shared memory are allocated per thread
        const unsigned sharedMemSize = blocks * 8 * sizeof(float);
        reg_spline_getJacobianValues3D_kernel<<<gridDims, blockDims, sharedMemSize>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                                                      controlPointImageDim, controlPointSpacing, referenceImageDim,
                                                                                      (unsigned)voxelNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_spline_getJacobianValues2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_getJacobianValues2D_kernel<<<gridDims, blockDims>>>(jacobianMatricesCuda, jacobianDetCuda, *controlPointTexture,
                                                                       controlPointImageDim, controlPointSpacing, referenceImageDim,
                                                                       (unsigned)voxelNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
double reg_spline_getJacobianPenaltyTerm_gpu(const nifti_image *referenceImage,
                                             const nifti_image *controlPointImage,
                                             const float4 *controlPointImageCuda,
                                             const bool& approx) {
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
        reg_spline_ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        jacSum = static_cast<double>(jacNumber);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        reg_spline_ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatricesCuda));

    // The Jacobian determinant are squared and logged (might not be english but will do)
    const unsigned blocks = CudaContext::GetBlockSize()->reg_spline_logSquaredValues;
    const unsigned grids = (unsigned)ceilf(sqrtf((float)jacNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_spline_logSquaredValues_kernel<<<gridDims, blockDims>>>(jacobianDetCuda, (unsigned)jacNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    // Perform the reduction
    const double penaltyTermValue = reg_sumReduction_gpu(jacobianDetCuda, jacNumber);
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDetCuda));
    return penaltyTermValue / jacSum;
}
/* *************************************************************** */
void reg_spline_getJacobianPenaltyTermGradient_gpu(const nifti_image *referenceImage,
                                                   const nifti_image *controlPointImage,
                                                   const float4 *controlPointImageCuda,
                                                   float4 *transGradientCuda,
                                                   const float& jacobianWeight,
                                                   const bool& approx) {
    auto blockSize = CudaContext::GetBlockSize();

    // The Jacobian matrices and determinants are computed
    float *jacobianMatricesCuda, *jacobianDetCuda;
    size_t jacNumber;
    if (approx) {
        jacNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        reg_spline_ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        // Allocate 3x3 matrices for 3D, and 2x2 matrices for 2D
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacNumber * sizeof(float)));
        reg_spline_ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }

    // Need to disorient the Jacobian matrix using the header information - voxel to real conversion
    const mat33 reorientation = reg_mat44_to_mat33(controlPointImage->sform_code > 0 ? &controlPointImage->sto_ijk : &controlPointImage->qto_ijk);

    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    const float3 weight = make_float3(referenceImage->dx * jacobianWeight / ((float)jacNumber * controlPointImage->dx),
                                      referenceImage->dy * jacobianWeight / ((float)jacNumber * controlPointImage->dy),
                                      referenceImage->dz * jacobianWeight / ((float)jacNumber * controlPointImage->dz));
    auto jacobianDeterminantTexture = Cuda::CreateTextureObject(jacobianDetCuda, cudaResourceTypeLinear, jacNumber * sizeof(float),
                                                                cudaChannelFormatKindFloat, 1);
    auto jacobianMatricesTexture = Cuda::CreateTextureObject(jacobianMatricesCuda, cudaResourceTypeLinear,
                                                             (controlPointImage->nz > 1 ? 9 : 4) * jacNumber * sizeof(float),
                                                             cudaChannelFormatKindFloat, 1);
    if (approx) {
        if (controlPointImage->nz > 1) {
            const unsigned blocks = blockSize->reg_spline_computeApproxJacGradient3D;
            const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            reg_spline_computeApproxJacGradient3D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                                                  *jacobianMatricesTexture, controlPointImageDim,
                                                                                  (unsigned)controlPointNumber, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        } else {
            const unsigned blocks = blockSize->reg_spline_computeApproxJacGradient2D;
            const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            reg_spline_computeApproxJacGradient2D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
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
            const unsigned blocks = blockSize->reg_spline_computeJacGradient3D;
            const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            reg_spline_computeJacGradient3D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
                                                                            *jacobianMatricesTexture, controlPointImageDim,
                                                                            controlPointVoxelSpacing, (unsigned)controlPointNumber,
                                                                            referenceImageDim, reorientation, weight);
            NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        } else {
            const unsigned blocks = blockSize->reg_spline_computeJacGradient2D;
            const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
            const dim3 gridDims(grids, grids, 1);
            const dim3 blockDims(blocks, 1, 1);
            reg_spline_computeJacGradient2D_kernel<<<gridDims, blockDims>>>(transGradientCuda, *jacobianDeterminantTexture,
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
double reg_spline_correctFolding_gpu(const nifti_image *referenceImage,
                                     const nifti_image *controlPointImage,
                                     float4 *controlPointImageCuda,
                                     const bool& approx) {
    auto blockSize = CudaContext::GetBlockSize();

    // The Jacobian matrices and determinants are computed
    float *jacobianMatricesCuda, *jacobianDetCuda;
    size_t jacobianDetSize, jacobianMatricesSize;
    size_t jacNumber; double jacSum;
    if (approx) {
        jacNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
        jacSum = (controlPointImage->nx - 2) * (controlPointImage->ny - 2) * (controlPointImage->nz - 2);
        jacobianDetSize = jacNumber * sizeof(float);
        jacobianMatricesSize = 9 * jacobianDetSize;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, jacobianMatricesSize));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacobianDetSize));
        reg_spline_ComputeApproxJacobianValues(controlPointImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    } else {
        jacNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
        jacSum = static_cast<double>(jacNumber);
        jacobianDetSize = jacNumber * sizeof(float);
        jacobianMatricesSize = 9 * jacobianDetSize;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatricesCuda, jacobianMatricesSize));
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDetCuda, jacobianDetSize));
        reg_spline_ComputeJacobianValues(controlPointImage, referenceImage, controlPointImageCuda, jacobianMatricesCuda, jacobianDetCuda);
    }

    // Check if the Jacobian determinant average
    float *jacobianDet2Cuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet2Cuda, jacobianDetSize));
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet2Cuda, jacobianDetCuda, jacobianDetSize, cudaMemcpyDeviceToDevice));
    const unsigned blocks = blockSize->reg_spline_logSquaredValues;
    const unsigned grids = (unsigned)ceilf(sqrtf((float)jacNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_spline_logSquaredValues_kernel<<<gridDims, blockDims>>>(jacobianDet2Cuda, (unsigned)jacNumber);
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
    auto jacobianDeterminantTexture = Cuda::CreateTextureObject(jacobianDetCuda, cudaResourceTypeLinear, jacobianDetSize,
                                                                cudaChannelFormatKindFloat, 1);
    auto jacobianMatricesTexture = Cuda::CreateTextureObject(jacobianMatricesCuda, cudaResourceTypeLinear, jacobianMatricesSize,
                                                             cudaChannelFormatKindFloat, 1);
    if (approx) {
        const unsigned blocks = blockSize->reg_spline_approxCorrectFolding3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_approxCorrectFolding3D_kernel<<<gridDims, blockDims>>>(controlPointImageCuda, *jacobianDeterminantTexture,
                                                                          *jacobianMatricesTexture, controlPointImageDim,
                                                                          controlPointSpacing, (unsigned)controlPointNumber, reorientation);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(controlPointImage->dx / referenceImage->dx,
                                                            controlPointImage->dy / referenceImage->dy,
                                                            controlPointImage->dz / referenceImage->dz);
        const unsigned blocks = blockSize->reg_spline_correctFolding3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)controlPointNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_spline_correctFolding3D_kernel<<<gridDims, blockDims>>>(controlPointImageCuda, *jacobianDeterminantTexture,
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
void reg_getDeformationFromDisplacement_gpu(const nifti_image *image, float4 *imageCuda, const bool& reverse = false) {
    // Bind the qform or sform
    const mat44 affineMatrix = image->sform_code > 0 ? image->sto_xyz : image->qto_xyz;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    const unsigned blocks = CudaContext::GetBlockSize()->reg_getDeformationFromDisplacement;
    const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_getDeformationFromDisplacement3D_kernel<<<gridDims, blockDims>>>(imageCuda, imageDim, (unsigned)voxelNumber, affineMatrix, reverse);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_getDisplacementFromDeformation_gpu(const nifti_image *image, float4 *imageCuda) {
    reg_getDeformationFromDisplacement_gpu(image, imageCuda, true);
}
/* *************************************************************** */
void reg_getDeformationFieldFromVelocityGrid_gpu(const nifti_image *controlPointImage,
                                                 const nifti_image *deformationField,
                                                 const float4 *controlPointImageCuda,
                                                 float4 *deformationFieldCuda) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);

    // Create a mask array where no voxel are excluded
    int *maskCuda = nullptr;
    NR_CUDA_SAFE_CALL(cudaMalloc(&maskCuda, voxelNumber * sizeof(int)));
    reg_fillMaskArray_gpu(maskCuda, voxelNumber);

    // Define some variables for the deformation fields
    float4 *tempDefCuda = nullptr;
    NR_CUDA_SAFE_CALL(cudaMalloc(&tempDefCuda, voxelNumber * sizeof(float4)));

    // The deformation field is computed
    reg_spline_getDeformationField_gpu(controlPointImage, deformationField, controlPointImageCuda,
                                       deformationFieldCuda, maskCuda, voxelNumber, true); // non-interpolant spline is used

    // The deformation field is converted into a displacement field
    reg_getDisplacementFromDeformation_gpu(deformationField, deformationFieldCuda);

    // Scaling of the deformation field
    const unsigned squaringNumber = (unsigned)fabs(controlPointImage->intent_p1);
    const float scalingValue = pow(2.f, (float)squaringNumber);
    // Backward/forward deformation field is scaled down
    reg_multiplyValue_gpu((int)voxelNumber, deformationFieldCuda, (controlPointImage->intent_p1 < 0  ? -1.f : 1.f) / scalingValue);

    // The displacement field is converted back into a deformation field
    reg_getDeformationFromDisplacement_gpu(deformationField, deformationFieldCuda);

    // The deformation field is squared
    for (unsigned i = 0; i < squaringNumber; ++i) {
        // The deformation field arrays are updated
        NR_CUDA_SAFE_CALL(cudaMemcpy(tempDefCuda, deformationFieldCuda, voxelNumber * sizeof(float4), cudaMemcpyDeviceToDevice));

        // The deformation fields are composed
        reg_defField_compose_gpu(deformationField, tempDefCuda, deformationFieldCuda, voxelNumber);
    }

    NR_CUDA_SAFE_CALL(cudaFree(tempDefCuda));
    NR_CUDA_SAFE_CALL(cudaFree(maskCuda));
}
/* *************************************************************** */
void reg_defField_compose_gpu(const nifti_image *deformationField,
                              const float4 *deformationFieldCuda,
                              float4 *deformationFieldCudaOut,
                              const size_t& activeVoxelNumber) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const int3 referenceImageDim = make_int3(deformationField->nx, deformationField->ny, deformationField->nz);
    const mat44 affineMatrixB = deformationField->sform_code > 0 ? deformationField->sto_ijk : deformationField->qto_ijk;
    const mat44 affineMatrixC = deformationField->sform_code > 0 ? deformationField->sto_xyz : deformationField->qto_xyz;
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, cudaResourceTypeLinear,
                                                             activeVoxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    if (deformationField->nz > 1) {
        const unsigned blocks = blockSize->reg_defField_compose3D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_defField_compose3D_kernel<<<gridDims, blockDims>>>(deformationFieldCudaOut, *deformationFieldTexture, referenceImageDim,
                                                               (unsigned)voxelNumber, affineMatrixB, affineMatrixC);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_defField_compose2D;
        const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_defField_compose2D_kernel<<<gridDims, blockDims>>>(deformationFieldCudaOut, *deformationFieldTexture, referenceImageDim,
                                                               (unsigned)voxelNumber, affineMatrixB, affineMatrixC);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_defField_getJacobianMatrix_gpu(const nifti_image *deformationField,
                                        const float4 *deformationFieldCuda,
                                        float *jacobianMatricesCuda) {
    const int3 referenceImageDim = make_int3(deformationField->nx, deformationField->ny, deformationField->nz);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const mat33 reorientation = reg_mat44_to_mat33(deformationField->sform_code > 0 ? &deformationField->sto_xyz : &deformationField->qto_xyz);
    auto deformationFieldTexture = Cuda::CreateTextureObject(deformationFieldCuda, cudaResourceTypeLinear,
                                                             voxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    const unsigned blocks = CudaContext::GetBlockSize()->reg_defField_getJacobianMatrix;
    const unsigned grids = (unsigned)ceilf(sqrtf((float)voxelNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_defField_getJacobianMatrix3D_kernel<<<gridDims, blockDims>>>(jacobianMatricesCuda, *deformationFieldTexture, referenceImageDim,
                                                                     (unsigned)voxelNumber, reorientation);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
