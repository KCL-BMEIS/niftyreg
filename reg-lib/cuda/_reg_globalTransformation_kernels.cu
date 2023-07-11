/*
 *  _reg_affineTransformation.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
__global__ void reg_affine_deformationField_kernel(float4 *deformationField,
                                                   const mat44 affineMatrix,
                                                   const int3 imageSize,
                                                   const unsigned voxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, imageSize.x * imageSize.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, imageSize.x, quot, rem);
        const int y = quot, x = rem;

        /* The transformation is applied */
        const float4 position = {
            affineMatrix.m[0][0] * x + affineMatrix.m[0][1] * y + affineMatrix.m[0][2] * z + affineMatrix.m[0][3],
            affineMatrix.m[1][0] * x + affineMatrix.m[1][1] * y + affineMatrix.m[1][2] * z + affineMatrix.m[1][3],
            affineMatrix.m[2][0] * x + affineMatrix.m[2][1] * y + affineMatrix.m[2][2] * z + affineMatrix.m[2][3],
            0.f
        };
        /* the deformation field (real coordinates) is stored */
        deformationField[tid] = position;
    }
}
/* *************************************************************** */
