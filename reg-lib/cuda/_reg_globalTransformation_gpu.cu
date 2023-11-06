/*
 *  _reg_globalTransformation_gpu.cu
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_globalTransformation_gpu.h"
#include "_reg_globalTransformation_kernels.cu"

/* *************************************************************** */
void reg_affine_getDeformationField_gpu(const mat44 *affineMatrix,
                                        const nifti_image *targetImage,
                                        float4 *deformationFieldCuda,
                                        const bool composition) {
    // TODO Implement composition
    if (composition)
        NR_FATAL_ERROR("Composition is not implemented on the GPU");

    const int3 imageSize = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    const size_t voxelNumber = targetImage->nvox;

    // If the target sform is defined, it is used. The qform is used otherwise
    const mat44 *targetMatrix = targetImage->sform_code > 0 ? &targetImage->sto_xyz : &targetImage->qto_xyz;

    // Affine * TargetMat * voxelIndex is performed
    // Affine * TargetMat is constant
    const mat44 transformationMatrix = reg_mat44_mul(affineMatrix, targetMatrix);

    const unsigned blocks = CudaContext::GetBlockSize()->reg_affine_getDeformationField;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)targetImage->nvox / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_affine_getDeformationField_kernel<<<gridDims, blockDims>>>(deformationFieldCuda, transformationMatrix, imageSize, (unsigned)voxelNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
