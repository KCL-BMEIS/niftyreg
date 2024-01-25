/*
 *  CudaGlobalTransformation.cu
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "CudaGlobalTransformation.hpp"
#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
template<bool is3d, bool compose>
void GetAffineDeformationField(const mat44 *affineMatrix,
                               const nifti_image *deformationField,
                               float4 *deformationFieldCuda) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, is3d ? 3 : 2);
    const int3 imageDims = make_int3(deformationField->nx, deformationField->ny, deformationField->nz);
    const mat44 *targetMatrix = deformationField->sform_code > 0 ? &deformationField->sto_xyz : &deformationField->qto_xyz;
    const mat44 transMatrix = compose ? *affineMatrix : reg_mat44_mul(affineMatrix, targetMatrix);
    Cuda::UniqueTextureObjectPtr deformationFieldTexturePtr; cudaTextureObject_t deformationFieldTexture = 0;
    if constexpr (compose) {
        deformationFieldTexturePtr = Cuda::CreateTextureObject(deformationFieldCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
        deformationFieldTexture = *deformationFieldTexturePtr;
    }

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [
        deformationFieldCuda, deformationFieldTexture, transMatrix, imageDims
    ]__device__(const int index) {
        float voxel[3];
        if constexpr (compose) {
            float4 defVal = tex1Dfetch<float4>(deformationFieldTexture, index);
            voxel[0] = defVal.x; voxel[1] = defVal.y; voxel[2] = defVal.z;
        } else {
            auto dims = reg_indexToDims_cuda<is3d>(index, imageDims);
            voxel[0] = static_cast<float>(dims.x);
            voxel[1] = static_cast<float>(dims.y);
            voxel[2] = static_cast<float>(dims.z);
        }

        // The transformation is applied
        float position[3];
        reg_mat44_mul_cuda<is3d>(transMatrix, voxel, position);

        // The deformation field (real coordinates) is stored
        deformationFieldCuda[index] = make_float4(position[0], position[1], position[2], 0);
    });
}
/* *************************************************************** */
template<bool compose>
void Cuda::GetAffineDeformationField(const mat44 *affineMatrix,
                                     const nifti_image *deformationField,
                                     float4 *deformationFieldCuda) {
    auto getAffineDeformationField = deformationField->nz > 1 ? ::GetAffineDeformationField<true, compose> :
                                                                ::GetAffineDeformationField<false, compose>;
    getAffineDeformationField(affineMatrix, deformationField, deformationFieldCuda);
}
template void Cuda::GetAffineDeformationField<false>(const mat44*, const nifti_image*, float4*);
template void Cuda::GetAffineDeformationField<true>(const mat44*, const nifti_image*, float4*);
/* *************************************************************** */
