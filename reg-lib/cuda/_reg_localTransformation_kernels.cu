/*
 *  _reg_localTransformation_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
__device__ void GetBasisBSplineValues(const float basis, float *values) {
    const float ff = Square(basis);
    const float fff = ff * basis;
    const float mf = 1.f - basis;
    values[0] = Cube(mf) / 6.f;
    values[1] = (3.f * fff - 6.f * ff + 4.f) / 6.f;
    values[2] = (-3.f * fff + 3.f * ff + 3.f * basis + 1.f) / 6.f;
    values[3] = fff / 6.f;
}
/* *************************************************************** */
__device__ void GetFirstBSplineValues(const float basis, float *values, float *first) {
    GetBasisBSplineValues(basis, values);
    first[3] = Square(basis) / 2.f;
    first[0] = basis - 0.5f - first[3];
    first[2] = 1.f + first[0] - 2.f * first[3];
    first[1] = -first[0] - first[2] - first[3];
}
/* *************************************************************** */
__device__ void GetBasisSplineValues(const float basis, float *values) {
    const float ff = Square(basis);
    values[0] = (basis * ((2.f - basis) * basis - 1.f)) / 2.f;
    values[1] = (ff * (3.f * basis - 5.f) + 2.f) / 2.f;
    values[2] = (basis * ((4.f - 3.f * basis) * basis + 1.f)) / 2.f;
    values[3] = (basis - 1.f) * ff / 2.f;
}
/* *************************************************************** */
__device__ void GetBasisSplineValuesX(const float basis, float4 *values) {
    const float ff = Square(basis);
    values->x = (basis * ((2.f - basis) * basis - 1.f)) / 2.f;
    values->y = (ff * (3.f * basis - 5.f) + 2.f) / 2.f;
    values->z = (basis * ((4.f - 3.f * basis) * basis + 1.f)) / 2.f;
    values->w = (basis - 1.f) * ff / 2.f;
}
/* *************************************************************** */
__device__ void GetBSplineBasisValue(const float basis, const int index, float *value, float *first) {
    switch (index) {
    case 0:
        *value = (1.f - basis) * (1.f - basis) * (1.f - basis) / 6.f;
        *first = (2.f * basis - basis * basis - 1.f) / 2.f;
        break;
    case 1:
        *value = (3.f * basis * basis * basis - 6.f * basis * basis + 4.f) / 6.f;
        *first = (3.f * basis * basis - 4.f * basis) / 2.f;
        break;
    case 2:
        *value = (3.f * basis * basis - 3.f * basis * basis * basis + 3.f * basis + 1.f) / 6.f;
        *first = (2.f * basis - 3.f * basis * basis + 1.f) / 2.f;
        break;
    case 3:
        *value = basis * basis * basis / 6.f;
        *first = basis * basis / 2.f;
        break;
    default:
        *value = 0.f;
        *first = 0.f;
        break;
    }
}
/* *************************************************************** */
__device__ void GetFirstDerivativeBasisValues2D(const int index, float *xBasis, float *yBasis) {
    switch (index) {
    case 0: xBasis[0] = -0.0833333f; yBasis[0] = -0.0833333f; break;
    case 1: xBasis[1] = 0.f; yBasis[1] = -0.333333f; break;
    case 2: xBasis[2] = 0.0833333f; yBasis[2] = -0.0833333f; break;
    case 3: xBasis[3] = -0.333333f; yBasis[3] = 0.f; break;
    case 4: xBasis[4] = 0.f; yBasis[4] = 0.f; break;
    case 5: xBasis[5] = 0.333333f; yBasis[5] = 0.f; break;
    case 6: xBasis[6] = -0.0833333f; yBasis[6] = 0.0833333f; break;
    case 7: xBasis[7] = 0.f; yBasis[7] = 0.333333f; break;
    case 8: xBasis[8] = 0.0833333f; yBasis[8] = 0.0833333f; break;
    }
}
/* *************************************************************** */
__device__ void GetFirstDerivativeBasisValues3D(const int index, float *xBasis, float *yBasis, float *zBasis) {
    switch (index) {
    case 0: xBasis[0] = -0.013889f; yBasis[0] = -0.013889f; zBasis[0] = -0.013889f; break;
    case 1: xBasis[1] = 0.000000f; yBasis[1] = -0.055556f; zBasis[1] = -0.055556f; break;
    case 2: xBasis[2] = 0.013889f; yBasis[2] = -0.013889f; zBasis[2] = -0.013889f; break;
    case 3: xBasis[3] = -0.055556f; yBasis[3] = 0.000000f; zBasis[3] = -0.055556f; break;
    case 4: xBasis[4] = 0.000000f; yBasis[4] = 0.000000f; zBasis[4] = -0.222222f; break;
    case 5: xBasis[5] = 0.055556f; yBasis[5] = 0.000000f; zBasis[5] = -0.055556f; break;
    case 6: xBasis[6] = -0.013889f; yBasis[6] = 0.013889f; zBasis[6] = -0.013889f; break;
    case 7: xBasis[7] = 0.000000f; yBasis[7] = 0.055556f; zBasis[7] = -0.055556f; break;
    case 8: xBasis[8] = 0.013889f; yBasis[8] = 0.013889f; zBasis[8] = -0.013889f; break;
    case 9: xBasis[9] = -0.055556f; yBasis[9] = -0.055556f; zBasis[9] = 0.000000f; break;
    case 10: xBasis[10] = 0.000000f; yBasis[10] = -0.222222f; zBasis[10] = 0.000000f; break;
    case 11: xBasis[11] = 0.055556f; yBasis[11] = -0.055556f; zBasis[11] = 0.000000f; break;
    case 12: xBasis[12] = -0.222222f; yBasis[12] = 0.000000f; zBasis[12] = 0.000000f; break;
    case 13: xBasis[13] = 0.000000f; yBasis[13] = 0.000000f; zBasis[13] = 0.000000f; break;
    case 14: xBasis[14] = 0.222222f; yBasis[14] = 0.000000f; zBasis[14] = 0.000000f; break;
    case 15: xBasis[15] = -0.055556f; yBasis[15] = 0.055556f; zBasis[15] = 0.000000f; break;
    case 16: xBasis[16] = 0.000000f; yBasis[16] = 0.222222f; zBasis[16] = 0.000000f; break;
    case 17: xBasis[17] = 0.055556f; yBasis[17] = 0.055556f; zBasis[17] = 0.000000f; break;
    case 18: xBasis[18] = -0.013889f; yBasis[18] = -0.013889f; zBasis[18] = 0.013889f; break;
    case 19: xBasis[19] = 0.000000f; yBasis[19] = -0.055556f; zBasis[19] = 0.055556f; break;
    case 20: xBasis[20] = 0.013889f; yBasis[20] = -0.013889f; zBasis[20] = 0.013889f; break;
    case 21: xBasis[21] = -0.055556f; yBasis[21] = 0.000000f; zBasis[21] = 0.055556f; break;
    case 22: xBasis[22] = 0.000000f; yBasis[22] = 0.000000f; zBasis[22] = 0.222222f; break;
    case 23: xBasis[23] = 0.055556f; yBasis[23] = 0.000000f; zBasis[23] = 0.055556f; break;
    case 24: xBasis[24] = -0.013889f; yBasis[24] = 0.013889f; zBasis[24] = 0.013889f; break;
    case 25: xBasis[25] = 0.000000f; yBasis[25] = 0.055556f; zBasis[25] = 0.055556f; break;
    case 26: xBasis[26] = 0.013889f; yBasis[26] = 0.013889f; zBasis[26] = 0.013889f; break;
    }
}
/* *************************************************************** */
__device__ void GetSecondDerivativeBasisValues2D(const int index, float *xxBasis, float *yyBasis, float *xyBasis) {
    switch (index) {
    case 0: xxBasis[0] = 0.166667f; yyBasis[0] = 0.166667f; xyBasis[0] = 0.25f; break;
    case 1: xxBasis[1] = -0.333333f; yyBasis[1] = 0.666667f; xyBasis[1] = -0.f; break;
    case 2: xxBasis[2] = 0.166667f; yyBasis[2] = 0.166667f; xyBasis[2] = -0.25f; break;
    case 3: xxBasis[3] = 0.666667f; yyBasis[3] = -0.333333f; xyBasis[3] = -0.f; break;
    case 4: xxBasis[4] = -1.33333f; yyBasis[4] = -1.33333f; xyBasis[4] = 0.f; break;
    case 5: xxBasis[5] = 0.666667f; yyBasis[5] = -0.333333f; xyBasis[5] = 0.f; break;
    case 6: xxBasis[6] = 0.166667f; yyBasis[6] = 0.166667f; xyBasis[6] = -0.25f; break;
    case 7: xxBasis[7] = -0.333333f; yyBasis[7] = 0.666667f; xyBasis[7] = 0.f; break;
    case 8: xxBasis[8] = 0.166667f; yyBasis[8] = 0.166667f; xyBasis[8] = 0.25f; break;
    }
}
/* *************************************************************** */
__device__ void GetSecondDerivativeBasisValues3D(const int index,
                                                 float *xxBasis,
                                                 float *yyBasis,
                                                 float *zzBasis,
                                                 float *xyBasis,
                                                 float *yzBasis,
                                                 float *xzBasis) {
    switch (index) {
    case 0:
        xxBasis[0] = 0.027778f; yyBasis[0] = 0.027778f; zzBasis[0] = 0.027778f;
        xyBasis[0] = 0.041667f; yzBasis[0] = 0.041667f; xzBasis[0] = 0.041667f;
        break;
    case 1:
        xxBasis[1] = -0.055556f; yyBasis[1] = 0.111111f; zzBasis[1] = 0.111111f;
        xyBasis[1] = -0.000000f; yzBasis[1] = 0.166667f; xzBasis[1] = -0.000000f;
        break;
    case 2:
        xxBasis[2] = 0.027778f; yyBasis[2] = 0.027778f; zzBasis[2] = 0.027778f;
        xyBasis[2] = -0.041667f; yzBasis[2] = 0.041667f; xzBasis[2] = -0.041667f;
        break;
    case 3:
        xxBasis[3] = 0.111111f; yyBasis[3] = -0.055556f; zzBasis[3] = 0.111111f;
        xyBasis[3] = -0.000000f; yzBasis[3] = -0.000000f; xzBasis[3] = 0.166667f;
        break;
    case 4:
        xxBasis[4] = -0.222222f; yyBasis[4] = -0.222222f; zzBasis[4] = 0.444444f;
        xyBasis[4] = 0.000000f; yzBasis[4] = -0.000000f; xzBasis[4] = -0.000000f;
        break;
    case 5:
        xxBasis[5] = 0.111111f; yyBasis[5] = -0.055556f; zzBasis[5] = 0.111111f;
        xyBasis[5] = 0.000000f; yzBasis[5] = -0.000000f; xzBasis[5] = -0.166667f;
        break;
    case 6:
        xxBasis[6] = 0.027778f; yyBasis[6] = 0.027778f; zzBasis[6] = 0.027778f;
        xyBasis[6] = -0.041667f; yzBasis[6] = -0.041667f; xzBasis[6] = 0.041667f;
        break;
    case 7:
        xxBasis[7] = -0.055556f; yyBasis[7] = 0.111111f; zzBasis[7] = 0.111111f;
        xyBasis[7] = 0.000000f; yzBasis[7] = -0.166667f; xzBasis[7] = -0.000000f;
        break;
    case 8:
        xxBasis[8] = 0.027778f; yyBasis[8] = 0.027778f; zzBasis[8] = 0.027778f;
        xyBasis[8] = 0.041667f; yzBasis[8] = -0.041667f; xzBasis[8] = -0.041667f;
        break;
    case 9:
        xxBasis[9] = 0.111111f; yyBasis[9] = 0.111111f; zzBasis[9] = -0.055556f;
        xyBasis[9] = 0.166667f; yzBasis[9] = -0.000000f; xzBasis[9] = -0.000000f;
        break;
    case 10:
        xxBasis[10] = -0.222222f; yyBasis[10] = 0.444444f; zzBasis[10] = -0.222222f;
        xyBasis[10] = -0.000000f; yzBasis[10] = -0.000000f; xzBasis[10] = 0.000000f;
        break;
    case 11:
        xxBasis[11] = 0.111111f; yyBasis[11] = 0.111111f; zzBasis[11] = -0.055556f;
        xyBasis[11] = -0.166667f; yzBasis[11] = -0.000000f; xzBasis[11] = 0.000000f;
        break;
    case 12:
        xxBasis[12] = 0.444444f; yyBasis[12] = -0.222222f; zzBasis[12] = -0.222222f;
        xyBasis[12] = -0.000000f; yzBasis[12] = 0.000000f; xzBasis[12] = -0.000000f;
        break;
    case 13:
        xxBasis[13] = -0.888889f; yyBasis[13] = -0.888889f; zzBasis[13] = -0.888889f;
        xyBasis[13] = 0.000000f; yzBasis[13] = 0.000000f; xzBasis[13] = 0.000000f;
        break;
    case 14:
        xxBasis[14] = 0.444444f; yyBasis[14] = -0.222222f; zzBasis[14] = -0.222222f;
        xyBasis[14] = 0.000000f; yzBasis[14] = 0.000000f; xzBasis[14] = 0.000000f;
        break;
    case 15:
        xxBasis[15] = 0.111111f; yyBasis[15] = 0.111111f; zzBasis[15] = -0.055556f;
        xyBasis[15] = -0.166667f; yzBasis[15] = 0.000000f; xzBasis[15] = -0.000000f;
        break;
    case 16:
        xxBasis[16] = -0.222222f; yyBasis[16] = 0.444444f; zzBasis[16] = -0.222222f;
        xyBasis[16] = 0.000000f; yzBasis[16] = 0.000000f; xzBasis[16] = 0.000000f;
        break;
    case 17:
        xxBasis[17] = 0.111111f; yyBasis[17] = 0.111111f; zzBasis[17] = -0.055556f;
        xyBasis[17] = 0.166667f; yzBasis[17] = 0.000000f; xzBasis[17] = 0.000000f;
        break;
    case 18:
        xxBasis[18] = 0.027778f; yyBasis[18] = 0.027778f; zzBasis[18] = 0.027778f;
        xyBasis[18] = 0.041667f; yzBasis[18] = -0.041667f; xzBasis[18] = -0.041667f;
        break;
    case 19:
        xxBasis[19] = -0.055556f; yyBasis[19] = 0.111111f; zzBasis[19] = 0.111111f;
        xyBasis[19] = -0.000000f; yzBasis[19] = -0.166667f; xzBasis[19] = 0.000000f;
        break;
    case 20:
        xxBasis[20] = 0.027778f; yyBasis[20] = 0.027778f; zzBasis[20] = 0.027778f;
        xyBasis[20] = -0.041667f; yzBasis[20] = -0.041667f; xzBasis[20] = 0.041667f;
        break;
    case 21:
        xxBasis[21] = 0.111111f; yyBasis[21] = -0.055556f; zzBasis[21] = 0.111111f;
        xyBasis[21] = -0.000000f; yzBasis[21] = 0.000000f; xzBasis[21] = -0.166667f;
        break;
    case 22:
        xxBasis[22] = -0.222222f; yyBasis[22] = -0.222222f; zzBasis[22] = 0.444444f;
        xyBasis[22] = 0.000000f; yzBasis[22] = 0.000000f; xzBasis[22] = 0.000000f;
        break;
    case 23:
        xxBasis[23] = 0.111111f; yyBasis[23] = -0.055556f; zzBasis[23] = 0.111111f;
        xyBasis[23] = 0.000000f; yzBasis[23] = 0.000000f; xzBasis[23] = 0.166667f;
        break;
    case 24:
        xxBasis[24] = 0.027778f; yyBasis[24] = 0.027778f; zzBasis[24] = 0.027778f;
        xyBasis[24] = -0.041667f; yzBasis[24] = 0.041667f; xzBasis[24] = -0.041667f;
        break;
    case 25:
        xxBasis[25] = -0.055556f; yyBasis[25] = 0.111111f; zzBasis[25] = 0.111111f;
        xyBasis[25] = 0.000000f; yzBasis[25] = 0.166667f; xzBasis[25] = 0.000000f;
        break;
    case 26:
        xxBasis[26] = 0.027778f; yyBasis[26] = 0.027778f; zzBasis[26] = 0.027778f;
        xyBasis[26] = 0.041667f; yzBasis[26] = 0.041667f; xzBasis[26] = 0.041667f;
        break;
    }
}
/* *************************************************************** */
__device__ float4 GetSlidedValues(int x, int y,
                                  cudaTextureObject_t deformationFieldTexture,
                                  const int3& referenceImageDim,
                                  const mat44& affineMatrix) {
    int newX = x;
    if (x < 0)
        newX = 0;
    else if (x >= referenceImageDim.x)
        newX = referenceImageDim.x - 1;

    int newY = y;
    if (y < 0)
        newY = 0;
    else if (y >= referenceImageDim.y)
        newY = referenceImageDim.y - 1;

    x -= newX;
    y -= newY;
    const float4& slidedValues = make_float4(x * affineMatrix.m[0][0] + y * affineMatrix.m[0][1],
                                             x * affineMatrix.m[1][0] + y * affineMatrix.m[1][1],
                                             0.f, 0.f);
    return slidedValues + tex1Dfetch<float4>(deformationFieldTexture, newY * referenceImageDim.x + newX);
}
/* *************************************************************** */
__device__ float4 GetSlidedValues(int x, int y, int z,
                                  cudaTextureObject_t deformationFieldTexture,
                                  const int3& referenceImageDim,
                                  const mat44& affineMatrix) {
    int newX = x;
    if (x < 0)
        newX = 0;
    else if (x >= referenceImageDim.x)
        newX = referenceImageDim.x - 1;

    int newY = y;
    if (y < 0)
        newY = 0;
    else if (y >= referenceImageDim.y)
        newY = referenceImageDim.y - 1;

    int newZ = z;
    if (z < 0)
        newZ = 0;
    else if (z >= referenceImageDim.z)
        newZ = referenceImageDim.z - 1;

    x -= newX;
    y -= newY;
    z -= newZ;
    const float4& slidedValues = make_float4(x * affineMatrix.m[0][0] + y * affineMatrix.m[0][1] + z * affineMatrix.m[0][2],
                                             x * affineMatrix.m[1][0] + y * affineMatrix.m[1][1] + z * affineMatrix.m[1][2],
                                             x * affineMatrix.m[2][0] + y * affineMatrix.m[2][1] + z * affineMatrix.m[2][2],
                                             0.f);
    return slidedValues + tex1Dfetch<float4>(deformationFieldTexture, (newZ * referenceImageDim.y + newY) * referenceImageDim.x + newX);
}
/* *************************************************************** */
__global__ void reg_spline_getDeformationField3D(float4 *deformationField,
                                                 cudaTextureObject_t controlPointTexture,
                                                 cudaTextureObject_t maskTexture,
                                                 const mat44 *realToVoxel,
                                                 const int3 referenceImageDim,
                                                 const int3 controlPointImageDim,
                                                 const float3 controlPointVoxelSpacing,
                                                 const unsigned activeVoxelNumber,
                                                 const bool composition,
                                                 const bool bspline) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= activeVoxelNumber) return;
    int3 nodePre;
    float3 basis;

    if (composition) { // Composition of deformation fields
        // The previous position at the current pixel position is read
        const float4 node = deformationField[tid];

        // From real to pixel position in the CPP
        const float xVoxel = (realToVoxel->m[0][0] * node.x +
                              realToVoxel->m[0][1] * node.y +
                              realToVoxel->m[0][2] * node.z +
                              realToVoxel->m[0][3]);
        const float yVoxel = (realToVoxel->m[1][0] * node.x +
                              realToVoxel->m[1][1] * node.y +
                              realToVoxel->m[1][2] * node.z +
                              realToVoxel->m[1][3]);
        const float zVoxel = (realToVoxel->m[2][0] * node.x +
                              realToVoxel->m[2][1] * node.y +
                              realToVoxel->m[2][2] * node.z +
                              realToVoxel->m[2][3]);

        if (xVoxel < 0 || xVoxel >= referenceImageDim.x ||
            yVoxel < 0 || yVoxel >= referenceImageDim.y ||
            zVoxel < 0 || zVoxel >= referenceImageDim.z) return;

        nodePre = { Floor(xVoxel), Floor(yVoxel), Floor(zVoxel) };
        basis = { xVoxel - float(nodePre.x--), yVoxel - float(nodePre.y--), zVoxel - float(nodePre.z--) };
    } else { // starting deformation field is blank - !composition
        const int tid2 = tex1Dfetch<int>(maskTexture, tid);
        const auto&& [x, y, z] = reg_indexToDims_cuda<true>(tid2, referenceImageDim);
        // The "nearest previous" node is determined [0,0,0]
        const float xVoxel = float(x) / controlPointVoxelSpacing.x;
        const float yVoxel = float(y) / controlPointVoxelSpacing.y;
        const float zVoxel = float(z) / controlPointVoxelSpacing.z;
        nodePre = { int(xVoxel), int(yVoxel), int(zVoxel) };
        basis = { xVoxel - float(nodePre.x), yVoxel - float(nodePre.y), zVoxel - float(nodePre.z) };
    }
    // Z basis values
    extern __shared__ float yBasis[];   // Shared memory
    const unsigned sharedMemIndex = 4 * threadIdx.x;
    // Compute the shared memory offset which corresponds to four times the number of threads per block
    float *zBasis = &yBasis[4 * blockDim.x * blockDim.y * blockDim.z];
    if (basis.z < 0) basis.z = 0; // rounding error
    if (bspline) GetBasisBSplineValues(basis.z, &zBasis[sharedMemIndex]);
    else GetBasisSplineValues(basis.z, &zBasis[sharedMemIndex]);

    // Y basis values
    if (basis.y < 0) basis.y = 0; // rounding error
    if (bspline) GetBasisBSplineValues(basis.y, &yBasis[sharedMemIndex]);
    else GetBasisSplineValues(basis.y, &yBasis[sharedMemIndex]);

    // X basis values
    float xBasis[4];
    if (basis.x < 0) basis.x = 0; // rounding error
    if (bspline) GetBasisBSplineValues(basis.x, xBasis);
    else GetBasisSplineValues(basis.x, xBasis);

    float4 displacement{};
    for (char c = 0; c < 4; c++) {
        int indexYZ = ((nodePre.z + c) * controlPointImageDim.y + nodePre.y) * controlPointImageDim.x;
        const float basisZ = zBasis[sharedMemIndex + c];
        for (char b = 0; b < 4; b++, indexYZ += controlPointImageDim.x) {
            int indexXYZ = indexYZ + nodePre.x;
            const float basisY = yBasis[sharedMemIndex + b];
            for (char a = 0; a < 4; a++, indexXYZ++) {
                const float4& nodeCoeff = tex1Dfetch<float4>(controlPointTexture, indexXYZ);
                const float xyzBasis = xBasis[a] * basisY * basisZ;
                displacement.x += xyzBasis * nodeCoeff.x;
                displacement.y += xyzBasis * nodeCoeff.y;
                displacement.z += xyzBasis * nodeCoeff.z;
            }
        }
    }
    deformationField[tid] = displacement;
}
/* *************************************************************** */
__global__ void reg_spline_getDeformationField2D(float4 *deformationField,
                                                 cudaTextureObject_t controlPointTexture,
                                                 cudaTextureObject_t maskTexture,
                                                 const mat44 *realToVoxel,
                                                 const int3 referenceImageDim,
                                                 const int3 controlPointImageDim,
                                                 const float3 controlPointVoxelSpacing,
                                                 const unsigned activeVoxelNumber,
                                                 const bool composition,
                                                 const bool bspline) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= activeVoxelNumber) return;
    int2 nodePre;
    float2 basis;

    if (composition) { // Composition of deformation fields
        // The previous position at the current pixel position is read
        const float4 node = deformationField[tid];

        // From real to pixel position in the CPP
        const float xVoxel = (realToVoxel->m[0][0] * node.x +
                              realToVoxel->m[0][1] * node.y +
                              realToVoxel->m[0][3]);
        const float yVoxel = (realToVoxel->m[1][0] * node.x +
                              realToVoxel->m[1][1] * node.y +
                              realToVoxel->m[1][3]);

        if (xVoxel < 0 || xVoxel >= referenceImageDim.x ||
            yVoxel < 0 || yVoxel >= referenceImageDim.y) return;

        nodePre = { Floor(xVoxel), Floor(yVoxel) };
        basis = { xVoxel - float(nodePre.x--), yVoxel - float(nodePre.y--) };
    } else { // starting deformation field is blank - !composition
        const int tid2 = tex1Dfetch<int>(maskTexture, tid);
        const auto&& [x, y, z] = reg_indexToDims_cuda<false>(tid2, referenceImageDim);
        // The "nearest previous" node is determined [0,0,0]
        const float xVoxel = float(x) / controlPointVoxelSpacing.x;
        const float yVoxel = float(y) / controlPointVoxelSpacing.y;
        nodePre = { int(xVoxel), int(yVoxel) };
        basis = { xVoxel - float(nodePre.x), yVoxel - float(nodePre.y) };
    }
    // Y basis values
    extern __shared__ float yBasis[]; // Shared memory
    const unsigned sharedMemIndex = 4 * threadIdx.x;
    if (basis.y < 0) basis.y = 0; // rounding error
    if (bspline) GetBasisBSplineValues(basis.y, &yBasis[sharedMemIndex]);
    else GetBasisSplineValues(basis.y, &yBasis[sharedMemIndex]);

    // X basis values
    float xBasis[4];
    if (basis.x < 0) basis.x = 0; // rounding error
    if (bspline) GetBasisBSplineValues(basis.x, xBasis);
    else GetBasisSplineValues(basis.x, xBasis);

    float4 displacement{};
    for (char b = 0; b < 4; b++) {
        int index = (nodePre.y + b) * controlPointImageDim.x + nodePre.x;
        const float basis = yBasis[sharedMemIndex + b];
        for (char a = 0; a < 4; a++, index++) {
            const float4& nodeCoeff = tex1Dfetch<float4>(controlPointTexture, index);
            const float xyBasis = xBasis[a] * basis;
            displacement.x += xyBasis * nodeCoeff.x;
            displacement.y += xyBasis * nodeCoeff.y;
        }
    }
    deformationField[tid] = displacement;
}
/* *************************************************************** */
__global__ void reg_spline_getApproxSecondDerivatives2D(float4 *secondDerivativeValues,
                                                        cudaTextureObject_t controlPointTexture,
                                                        const int3 controlPointImageDim,
                                                        const unsigned controlPointNumber) {
    __shared__ float xxbasis[9];
    __shared__ float yybasis[9];
    __shared__ float xybasis[9];

    if (threadIdx.x < 9)
        GetSecondDerivativeBasisValues2D(threadIdx.x, xxbasis, yybasis, xybasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float4 xx{}, yy{}, xy{};
        unsigned tempIndex;
        if (0 < x && x < controlPointImageDim.x - 1 && 0 < y && y < controlPointImageDim.y - 1) {
            tempIndex = 0;
            for (int b = y - 1; b < y + 2; ++b) {
                for (int a = x - 1; a < x + 2; ++a) {
                    const int indexXY = b * controlPointImageDim.x + a;
                    const float4 controlPointValues = tex1Dfetch<float4>(controlPointTexture, indexXY);
                    xx.x += xxbasis[tempIndex] * controlPointValues.x;
                    xx.y += xxbasis[tempIndex] * controlPointValues.y;
                    yy.x += yybasis[tempIndex] * controlPointValues.x;
                    yy.y += yybasis[tempIndex] * controlPointValues.y;
                    xy.x += xybasis[tempIndex] * controlPointValues.x;
                    xy.y += xybasis[tempIndex] * controlPointValues.y;
                    tempIndex++;
                }
            }
        }

        tempIndex = 3 * tid;
        secondDerivativeValues[tempIndex++] = xx;
        secondDerivativeValues[tempIndex++] = yy;
        secondDerivativeValues[tempIndex] = xy;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxSecondDerivatives3D(float4 *secondDerivativeValues,
                                                        cudaTextureObject_t controlPointTexture,
                                                        const int3 controlPointImageDim,
                                                        const unsigned controlPointNumber) {
    __shared__ float xxbasis[27];
    __shared__ float yybasis[27];
    __shared__ float zzbasis[27];
    __shared__ float xybasis[27];
    __shared__ float yzbasis[27];
    __shared__ float xzbasis[27];

    if (threadIdx.x < 27)
        GetSecondDerivativeBasisValues3D(threadIdx.x, xxbasis, yybasis, zzbasis, xybasis, yzbasis, xzbasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int tempIndex = tid;
        int quot, rem;
        reg_div_cuda(tempIndex, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float4 xx{}, yy{}, zz{}, xy{}, yz{}, xz{};
        if (0 < x && x < controlPointImageDim.x - 1 && 0 < y && y < controlPointImageDim.y - 1 && 0 < z && z < controlPointImageDim.z - 1) {
            tempIndex = 0;
            for (int c = z - 1; c < z + 2; ++c) {
                for (int b = y - 1; b < y + 2; ++b) {
                    for (int a = x - 1; a < x + 2; ++a) {
                        const int indexXYZ = (c * controlPointImageDim.y + b) * controlPointImageDim.x + a;
                        const float4 controlPointValues = tex1Dfetch<float4>(controlPointTexture, indexXYZ);
                        xx = xx + xxbasis[tempIndex] * controlPointValues;
                        yy = yy + yybasis[tempIndex] * controlPointValues;
                        zz = zz + zzbasis[tempIndex] * controlPointValues;
                        xy = xy + xybasis[tempIndex] * controlPointValues;
                        yz = yz + yzbasis[tempIndex] * controlPointValues;
                        xz = xz + xzbasis[tempIndex] * controlPointValues;
                        tempIndex++;
                    }
                }
            }
        }

        tempIndex = 6 * tid;
        secondDerivativeValues[tempIndex++] = xx;
        secondDerivativeValues[tempIndex++] = yy;
        secondDerivativeValues[tempIndex++] = zz;
        secondDerivativeValues[tempIndex++] = xy;
        secondDerivativeValues[tempIndex++] = yz;
        secondDerivativeValues[tempIndex] = xz;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergy2D_kernel(float *penaltyTerm,
                                                           cudaTextureObject_t secondDerivativesTexture,
                                                           const unsigned controlPointNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        unsigned index = tid * 3;
        float4 xx = tex1Dfetch<float4>(secondDerivativesTexture, index++);  xx = xx * xx;
        float4 yy = tex1Dfetch<float4>(secondDerivativesTexture, index++);  yy = yy * yy;
        float4 xy = tex1Dfetch<float4>(secondDerivativesTexture, index++);  xy = xy * xy;
        penaltyTerm[tid] = xx.x + xx.y + yy.x + yy.y + 2.f * (xy.x + xy.y);
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergy3D_kernel(float *penaltyTerm,
                                                           cudaTextureObject_t secondDerivativesTexture,
                                                           const unsigned controlPointNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        unsigned index = tid * 6;
        float4 xx = tex1Dfetch<float4>(secondDerivativesTexture, index++);  xx = xx * xx;
        float4 yy = tex1Dfetch<float4>(secondDerivativesTexture, index++);  yy = yy * yy;
        float4 zz = tex1Dfetch<float4>(secondDerivativesTexture, index++);  zz = zz * zz;
        float4 xy = tex1Dfetch<float4>(secondDerivativesTexture, index++);  xy = xy * xy;
        float4 yz = tex1Dfetch<float4>(secondDerivativesTexture, index++);  yz = yz * yz;
        float4 xz = tex1Dfetch<float4>(secondDerivativesTexture, index);    xz = xz * xz;
        penaltyTerm[tid] = xx.x + xx.y + xx.z + yy.x + yy.y + yy.z + zz.x + zz.y + zz.z +
            2.f * (xy.x + xy.y + xy.z + yz.x + yz.y + yz.z + xz.x + xz.y + xz.z);
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergyGradient2D_kernel(float4 *nodeGradient,
                                                                   cudaTextureObject_t secondDerivativesTexture,
                                                                   const int3 controlPointImageDim,
                                                                   const unsigned controlPointNumber,
                                                                   const float weight) {
    __shared__ float xxbasis[9];
    __shared__ float yybasis[9];
    __shared__ float xybasis[9];

    if (threadIdx.x < 9)
        GetSecondDerivativeBasisValues2D(threadIdx.x, xxbasis, yybasis, xybasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float2 gradientValue{};
        float4 secondDerivativeValues;
        int coord = 0;
        for (int b = y - 1; b < y + 2; ++b) {
            for (int a = x - 1; a < x + 2; ++a) {
                if (-1 < a && a < controlPointImageDim.x && -1 < b && b < controlPointImageDim.y) {
                    int indexXY = 3 * (b * controlPointImageDim.x + a);
                    secondDerivativeValues = tex1Dfetch<float4>(secondDerivativesTexture, indexXY++); // XX
                    gradientValue.x += secondDerivativeValues.x * xxbasis[coord];
                    gradientValue.y += secondDerivativeValues.y * xxbasis[coord];
                    secondDerivativeValues = tex1Dfetch<float4>(secondDerivativesTexture, indexXY++); // YY
                    gradientValue.x += secondDerivativeValues.x * yybasis[coord];
                    gradientValue.y += secondDerivativeValues.y * yybasis[coord];
                    secondDerivativeValues = 2.f * tex1Dfetch<float4>(secondDerivativesTexture, indexXY); // XY
                    gradientValue.x += secondDerivativeValues.x * xybasis[coord];
                    gradientValue.y += secondDerivativeValues.y * xybasis[coord];
                }
                coord++;
            }
        }

        nodeGradient[tid].x += weight * gradientValue.x;
        nodeGradient[tid].y += weight * gradientValue.y;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergyGradient3D_kernel(float4 *nodeGradient,
                                                                   cudaTextureObject_t secondDerivativesTexture,
                                                                   const int3 controlPointImageDim,
                                                                   const unsigned controlPointNumber,
                                                                   const float weight) {
    __shared__ float xxbasis[27];
    __shared__ float yybasis[27];
    __shared__ float zzbasis[27];
    __shared__ float xybasis[27];
    __shared__ float yzbasis[27];
    __shared__ float xzbasis[27];

    if (threadIdx.x < 27)
        GetSecondDerivativeBasisValues3D(threadIdx.x, xxbasis, yybasis, zzbasis, xybasis, yzbasis, xzbasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float3 gradientValue{};
        float4 secondDerivativeValues;
        int coord = 0;
        for (int c = z - 1; c < z + 2; ++c) {
            for (int b = y - 1; b < y + 2; ++b) {
                for (int a = x - 1; a < x + 2; ++a) {
                    if (-1 < a && a < controlPointImageDim.x && -1 < b && b < controlPointImageDim.y && -1 < c && c < controlPointImageDim.z) {
                        unsigned indexXYZ = 6 * ((c * controlPointImageDim.y + b) * controlPointImageDim.x + a);
                        secondDerivativeValues = tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++); // XX
                        gradientValue.x += secondDerivativeValues.x * xxbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xxbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xxbasis[coord];
                        secondDerivativeValues = tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++); // YY
                        gradientValue.x += secondDerivativeValues.x * yybasis[coord];
                        gradientValue.y += secondDerivativeValues.y * yybasis[coord];
                        gradientValue.z += secondDerivativeValues.z * yybasis[coord];
                        secondDerivativeValues = tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++); // ZZ
                        gradientValue.x += secondDerivativeValues.x * zzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * zzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * zzbasis[coord];
                        secondDerivativeValues = 2.f * tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++); // XY
                        gradientValue.x += secondDerivativeValues.x * xybasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xybasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xybasis[coord];
                        secondDerivativeValues = 2.f * tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ++); // YZ
                        gradientValue.x += secondDerivativeValues.x * yzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * yzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * yzbasis[coord];
                        secondDerivativeValues = 2.f * tex1Dfetch<float4>(secondDerivativesTexture, indexXYZ); // XZ
                        gradientValue.x += secondDerivativeValues.x * xzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xzbasis[coord];
                    }
                    coord++;
                }
            }
        }
        gradientValue = weight * gradientValue;

        float4 metricGradientValue = nodeGradient[tid];
        metricGradientValue.x += gradientValue.x;
        metricGradientValue.y += gradientValue.y;
        metricGradientValue.z += gradientValue.z;
        nodeGradient[tid] = metricGradientValue;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxJacobianValues2D_kernel(float *jacobianMatrices,
                                                            float *jacobianDet,
                                                            cudaTextureObject_t controlPointTexture,
                                                            const int3 controlPointImageDim,
                                                            const unsigned controlPointNumber,
                                                            const mat33 reorientation) {
    __shared__ float xbasis[9];
    __shared__ float ybasis[9];

    if (threadIdx.x < 9)
        GetFirstDerivativeBasisValues2D(threadIdx.x, xbasis, ybasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        if (0 < x && x < controlPointImageDim.x - 1 && 0 < y && y < controlPointImageDim.y - 1) {
            float2 tx{}, ty{};
            unsigned index = 0;
            for (int b = y - 1; b < y + 2; ++b) {
                for (int a = x - 1; a < x + 2; ++a) {
                    const int indexXY = b * controlPointImageDim.x + a;
                    const float4 controlPointValues = tex1Dfetch<float4>(controlPointTexture, indexXY);
                    tx.x += xbasis[index] * controlPointValues.x;
                    tx.y += ybasis[index] * controlPointValues.x;
                    ty.x += xbasis[index] * controlPointValues.y;
                    ty.y += ybasis[index] * controlPointValues.y;
                    index++;
                }
            }

            // The jacobian matrix is reoriented
            float2 tx2, ty2;
            tx2.x = reorientation.m[0][0] * tx.x + reorientation.m[0][1] * ty.x;
            tx2.y = reorientation.m[0][0] * tx.y + reorientation.m[0][1] * ty.y;
            ty2.x = reorientation.m[1][0] * tx.x + reorientation.m[1][1] * ty.x;
            ty2.y = reorientation.m[1][0] * tx.y + reorientation.m[1][1] * ty.y;

            // The Jacobian matrix is stored
            index = tid * 4;
            jacobianMatrices[index++] = tx2.x;
            jacobianMatrices[index++] = tx2.y;
            jacobianMatrices[index++] = ty2.x;
            jacobianMatrices[index] = ty2.y;

            // The Jacobian determinant is computed and stored
            jacobianDet[tid] = tx2.x * ty2.y - tx2.y * ty2.x;
        } else {
            unsigned index = tid * 4;
            jacobianMatrices[index++] = 1.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index] = 1.f;
            jacobianDet[tid] = 1.f;
        }
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxJacobianValues3D_kernel(float *jacobianMatrices,
                                                            float *jacobianDet,
                                                            cudaTextureObject_t controlPointTexture,
                                                            const int3 controlPointImageDim,
                                                            const unsigned controlPointNumber,
                                                            const mat33 reorientation) {
    __shared__ float xbasis[27];
    __shared__ float ybasis[27];
    __shared__ float zbasis[27];

    if (threadIdx.x < 27)
        GetFirstDerivativeBasisValues3D(threadIdx.x, xbasis, ybasis, zbasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        if (0 < x && x < controlPointImageDim.x - 1 && 0 < y && y < controlPointImageDim.y - 1 && 0 < z && z < controlPointImageDim.z - 1) {
            float3 tx{}, ty{}, tz{};
            unsigned index = 0;
            for (int c = z - 1; c < z + 2; ++c) {
                for (int b = y - 1; b < y + 2; ++b) {
                    for (int a = x - 1; a < x + 2; ++a) {
                        const int indexXYZ = (c * controlPointImageDim.y + b) * controlPointImageDim.x + a;
                        const float4 controlPointValues = tex1Dfetch<float4>(controlPointTexture, indexXYZ);
                        tx.x += xbasis[index] * controlPointValues.x;
                        tx.y += ybasis[index] * controlPointValues.x;
                        tx.z += zbasis[index] * controlPointValues.x;
                        ty.x += xbasis[index] * controlPointValues.y;
                        ty.y += ybasis[index] * controlPointValues.y;
                        ty.z += zbasis[index] * controlPointValues.y;
                        tz.x += xbasis[index] * controlPointValues.z;
                        tz.y += ybasis[index] * controlPointValues.z;
                        tz.z += zbasis[index] * controlPointValues.z;
                        index++;
                    }
                }
            }

            // The jacobian matrix is reoriented
            float3 tx2, ty2, tz2;
            tx2.x = reorientation.m[0][0] * tx.x + reorientation.m[0][1] * ty.x + reorientation.m[0][2] * tz.x;
            tx2.y = reorientation.m[0][0] * tx.y + reorientation.m[0][1] * ty.y + reorientation.m[0][2] * tz.y;
            tx2.z = reorientation.m[0][0] * tx.z + reorientation.m[0][1] * ty.z + reorientation.m[0][2] * tz.z;
            ty2.x = reorientation.m[1][0] * tx.x + reorientation.m[1][1] * ty.x + reorientation.m[1][2] * tz.x;
            ty2.y = reorientation.m[1][0] * tx.y + reorientation.m[1][1] * ty.y + reorientation.m[1][2] * tz.y;
            ty2.z = reorientation.m[1][0] * tx.z + reorientation.m[1][1] * ty.z + reorientation.m[1][2] * tz.z;
            tz2.x = reorientation.m[2][0] * tx.x + reorientation.m[2][1] * ty.x + reorientation.m[2][2] * tz.x;
            tz2.y = reorientation.m[2][0] * tx.y + reorientation.m[2][1] * ty.y + reorientation.m[2][2] * tz.y;
            tz2.z = reorientation.m[2][0] * tx.z + reorientation.m[2][1] * ty.z + reorientation.m[2][2] * tz.z;

            // The Jacobian matrix is stored
            index = tid * 9;
            jacobianMatrices[index++] = tx2.x;
            jacobianMatrices[index++] = tx2.y;
            jacobianMatrices[index++] = tx2.z;
            jacobianMatrices[index++] = ty2.x;
            jacobianMatrices[index++] = ty2.y;
            jacobianMatrices[index++] = ty2.z;
            jacobianMatrices[index++] = tz2.x;
            jacobianMatrices[index++] = tz2.y;
            jacobianMatrices[index] = tz2.z;

            // The Jacobian determinant is computed and stored
            jacobianDet[tid] = tx2.x * ty2.y * tz2.z
                + tx2.y * ty2.z * tz2.x
                + tx2.z * ty2.x * tz2.y
                - tx2.x * ty2.z * tz2.y
                - tx2.y * ty2.x * tz2.z
                - tx2.z * ty2.y * tz2.x;
        } else {
            unsigned index = tid * 9;
            jacobianMatrices[index++] = 1.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 1.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index++] = 0.f;
            jacobianMatrices[index] = 1.f;
            jacobianDet[tid] = 1.f;
        }
    }
}
/* *************************************************************** */
__global__ void reg_spline_getJacobianValues2D_kernel(float *jacobianMatrices,
                                                      float *jacobianDet,
                                                      cudaTextureObject_t controlPointTexture,
                                                      const int3 controlPointImageDim,
                                                      const float3 controlPointSpacing,
                                                      const int3 referenceImageDim,
                                                      const unsigned voxelNumber,
                                                      const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        // the "nearest previous" node is determined [0,0,0]
        const int2 nodePre = { Floor((float)x / controlPointSpacing.x), Floor((float)y / controlPointSpacing.y) };

        float xBasis[4], yBasis[4], xFirst[4], yFirst[4], relative;

        relative = fabsf((float)x / controlPointSpacing.x - (float)nodePre.x);
        GetFirstBSplineValues(relative, xBasis, xFirst);

        relative = fabsf((float)y / controlPointSpacing.y - (float)nodePre.y);
        GetFirstBSplineValues(relative, yBasis, yFirst);

        float2 tx{}, ty{};
        for (int b = 0; b < 4; ++b) {
            int indexXY = (nodePre.y + b) * controlPointImageDim.x + nodePre.x;

            float4 nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXY++);
            float2 basis = make_float2(xFirst[0] * yBasis[b], xBasis[0] * yFirst[b]);
            tx = tx + nodeCoefficient.x * basis;
            ty = ty + nodeCoefficient.y * basis;

            nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXY++);
            basis = make_float2(xFirst[1] * yBasis[b], xBasis[1] * yFirst[b]);
            tx = tx + nodeCoefficient.x * basis;
            ty = ty + nodeCoefficient.y * basis;

            nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXY++);
            basis = make_float2(xFirst[2] * yBasis[b], xBasis[2] * yFirst[b]);
            tx = tx + nodeCoefficient.x * basis;
            ty = ty + nodeCoefficient.y * basis;

            nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXY);
            basis = make_float2(xFirst[3] * yBasis[b], xBasis[3] * yFirst[b]);
            tx = tx + nodeCoefficient.x * basis;
            ty = ty + nodeCoefficient.y * basis;
        }

        // The jacobian matrix is reoriented
        float2 tx2, ty2;
        tx2.x = reorientation.m[0][0] * tx.x + reorientation.m[0][1] * ty.x;
        tx2.y = reorientation.m[0][0] * tx.y + reorientation.m[0][1] * ty.y;
        ty2.x = reorientation.m[1][0] * tx.x + reorientation.m[1][1] * ty.x;
        ty2.y = reorientation.m[1][0] * tx.y + reorientation.m[1][1] * ty.y;

        // The Jacobian matrix is stored
        unsigned index = tid * 4;
        jacobianMatrices[index++] = tx2.x;
        jacobianMatrices[index++] = tx2.y;
        jacobianMatrices[index++] = ty2.x;
        jacobianMatrices[index] = ty2.y;

        // The Jacobian determinant is computed and stored
        jacobianDet[tid] = tx2.x * ty2.y - tx2.y * ty2.x;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getJacobianValues3D_kernel(float *jacobianMatrices,
                                                      float *jacobianDet,
                                                      cudaTextureObject_t controlPointTexture,
                                                      const int3 controlPointImageDim,
                                                      const float3 controlPointSpacing,
                                                      const int3 referenceImageDim,
                                                      const unsigned voxelNumber,
                                                      const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, referenceImageDim.x * referenceImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        // the "nearest previous" node is determined [0,0,0]
        const int3 nodePre = {
            Floor((float)x / controlPointSpacing.x),
            Floor((float)y / controlPointSpacing.y),
            Floor((float)z / controlPointSpacing.z)
        };

        extern __shared__ float yFirst[];
        float *zFirst = &yFirst[4 * blockDim.x * blockDim.y * blockDim.z];

        float xBasis[4], yBasis[4], zBasis[4], xFirst[4], relative;
        const unsigned sharedMemIndex = 4 * threadIdx.x;

        relative = fabsf((float)x / controlPointSpacing.x - (float)nodePre.x);
        GetFirstBSplineValues(relative, xBasis, xFirst);

        relative = fabsf((float)y / controlPointSpacing.y - (float)nodePre.y);
        GetFirstBSplineValues(relative, yBasis, &yFirst[sharedMemIndex]);

        relative = fabsf((float)z / controlPointSpacing.z - (float)nodePre.z);
        GetFirstBSplineValues(relative, zBasis, &zFirst[sharedMemIndex]);

        float3 tx{}, ty{}, tz{};
        for (int c = 0; c < 4; ++c) {
            for (int b = 0; b < 4; ++b) {
                int indexXYZ = ((nodePre.z + c) * controlPointImageDim.y + nodePre.y + b) * controlPointImageDim.x + nodePre.x;
                float3 basisXY{ yBasis[b] * zBasis[c], yFirst[sharedMemIndex + b] * zBasis[c], yBasis[b] * zFirst[sharedMemIndex + c] };

                float4 nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXYZ++);
                float3 basis = make_float3(xFirst[0], xBasis[0], xBasis[0]) * basisXY;
                tx = tx + nodeCoefficient.x * basis;
                ty = ty + nodeCoefficient.y * basis;
                tz = tz + nodeCoefficient.z * basis;

                nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXYZ++);
                basis = make_float3(xFirst[1], xBasis[1], xBasis[1]) * basisXY;
                tx = tx + nodeCoefficient.x * basis;
                ty = ty + nodeCoefficient.y * basis;
                tz = tz + nodeCoefficient.z * basis;

                nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXYZ++);
                basis = make_float3(xFirst[2], xBasis[2], xBasis[2]) * basisXY;
                tx = tx + nodeCoefficient.x * basis;
                ty = ty + nodeCoefficient.y * basis;
                tz = tz + nodeCoefficient.z * basis;

                nodeCoefficient = tex1Dfetch<float4>(controlPointTexture, indexXYZ);
                basis = make_float3(xFirst[3], xBasis[3], xBasis[3]) * basisXY;
                tx = tx + nodeCoefficient.x * basis;
                ty = ty + nodeCoefficient.y * basis;
                tz = tz + nodeCoefficient.z * basis;
            }
        }

        // The jacobian matrix is reoriented
        float3 tx2, ty2, tz2;
        tx2.x = reorientation.m[0][0] * tx.x + reorientation.m[0][1] * ty.x + reorientation.m[0][2] * tz.x;
        tx2.y = reorientation.m[0][0] * tx.y + reorientation.m[0][1] * ty.y + reorientation.m[0][2] * tz.y;
        tx2.z = reorientation.m[0][0] * tx.z + reorientation.m[0][1] * ty.z + reorientation.m[0][2] * tz.z;
        ty2.x = reorientation.m[1][0] * tx.x + reorientation.m[1][1] * ty.x + reorientation.m[1][2] * tz.x;
        ty2.y = reorientation.m[1][0] * tx.y + reorientation.m[1][1] * ty.y + reorientation.m[1][2] * tz.y;
        ty2.z = reorientation.m[1][0] * tx.z + reorientation.m[1][1] * ty.z + reorientation.m[1][2] * tz.z;
        tz2.x = reorientation.m[2][0] * tx.x + reorientation.m[2][1] * ty.x + reorientation.m[2][2] * tz.x;
        tz2.y = reorientation.m[2][0] * tx.y + reorientation.m[2][1] * ty.y + reorientation.m[2][2] * tz.y;
        tz2.z = reorientation.m[2][0] * tx.z + reorientation.m[2][1] * ty.z + reorientation.m[2][2] * tz.z;

        // The Jacobian matrix is stored
        unsigned index = tid * 9;
        jacobianMatrices[index++] = tx2.x;
        jacobianMatrices[index++] = tx2.y;
        jacobianMatrices[index++] = tx2.z;
        jacobianMatrices[index++] = ty2.x;
        jacobianMatrices[index++] = ty2.y;
        jacobianMatrices[index++] = ty2.z;
        jacobianMatrices[index++] = tz2.x;
        jacobianMatrices[index++] = tz2.y;
        jacobianMatrices[index] = tz2.z;

        // The Jacobian determinant is computed and stored
        jacobianDet[tid] = tx2.x * ty2.y * tz2.z
            + tx2.y * ty2.z * tz2.x
            + tx2.z * ty2.x * tz2.y
            - tx2.x * ty2.z * tz2.y
            - tx2.y * ty2.x * tz2.z
            - tx2.z * ty2.y * tz2.x;
    }
}
/* *************************************************************** */
__global__ void reg_spline_logSquaredValues_kernel(float *det, const unsigned voxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        const float val = logf(det[tid]);
        det[tid] = val * val;
    }
}
/* *************************************************************** */
__device__ void GetJacobianGradientValues2D(float *jacobianMatrix,
                                            float detJac,
                                            float basisX,
                                            float basisY,
                                            float2 *jacobianConstraint) {
    jacobianConstraint->x += detJac * (basisX * jacobianMatrix[3] - basisY * jacobianMatrix[2]);
    jacobianConstraint->y += detJac * (basisY * jacobianMatrix[0] - basisX * jacobianMatrix[1]);
}
/* *************************************************************** */
__device__ void GetJacobianGradientValues3D(float *jacobianMatrix,
                                            float detJac,
                                            float basisX,
                                            float basisY,
                                            float basisZ,
                                            float3 *jacobianConstraint) {
    jacobianConstraint->x += detJac * (
        basisX * (jacobianMatrix[4] * jacobianMatrix[8] - jacobianMatrix[5] * jacobianMatrix[7]) +
        basisY * (jacobianMatrix[5] * jacobianMatrix[6] - jacobianMatrix[3] * jacobianMatrix[8]) +
        basisZ * (jacobianMatrix[3] * jacobianMatrix[7] - jacobianMatrix[4] * jacobianMatrix[6]));

    jacobianConstraint->y += detJac * (
        basisX * (jacobianMatrix[2] * jacobianMatrix[7] - jacobianMatrix[1] * jacobianMatrix[8]) +
        basisY * (jacobianMatrix[0] * jacobianMatrix[8] - jacobianMatrix[2] * jacobianMatrix[6]) +
        basisZ * (jacobianMatrix[1] * jacobianMatrix[6] - jacobianMatrix[0] * jacobianMatrix[7]));

    jacobianConstraint->z += detJac * (
        basisX * (jacobianMatrix[1] * jacobianMatrix[5] - jacobianMatrix[2] * jacobianMatrix[4]) +
        basisY * (jacobianMatrix[2] * jacobianMatrix[3] - jacobianMatrix[0] * jacobianMatrix[5]) +
        basisZ * (jacobianMatrix[0] * jacobianMatrix[4] - jacobianMatrix[1] * jacobianMatrix[3]));
}
/* *************************************************************** */
__global__ void reg_spline_computeApproxJacGradient2D_kernel(float4 *gradient,
                                                             cudaTextureObject_t jacobianDeterminantTexture,
                                                             cudaTextureObject_t jacobianMatricesTexture,
                                                             const int3 controlPointImageDim,
                                                             const unsigned controlPointNumber,
                                                             const mat33 reorientation,
                                                             const float3 weight) {
    __shared__ float xbasis[9];
    __shared__ float ybasis[9];

    if (threadIdx.x < 9)
        GetFirstDerivativeBasisValues2D(threadIdx.x, xbasis, ybasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float2 jacobianGradient{};
        unsigned index = 8;
        for (int pixelY = y - 1; pixelY < y + 2; ++pixelY) {
            if (0 < pixelY && pixelY < controlPointImageDim.y - 1) {
                int jacIndex = pixelY * controlPointImageDim.x + x - 1;
                for (int pixelX = (int)(x - 1); pixelX < (int)(x + 2); ++pixelX) {
                    if (0 < pixelX && pixelX < controlPointImageDim.x - 1) {
                        float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);
                        if (detJac > 0.f) {
                            detJac = 2.f * logf(detJac) / detJac;
                            float jacobianMatrix[4];
                            jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 4);
                            jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 4 + 1);
                            jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 4 + 2);
                            jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 4 + 3);
                            GetJacobianGradientValues2D(jacobianMatrix, detJac, xbasis[index], ybasis[index], &jacobianGradient);
                        }
                    }
                    jacIndex++;
                    index--;
                }
            } else index -= 3;
        }

        gradient[tid] = gradient[tid] + make_float4(
            weight.x * (reorientation.m[0][0] * jacobianGradient.x + reorientation.m[0][1] * jacobianGradient.y),
            weight.y * (reorientation.m[1][0] * jacobianGradient.x + reorientation.m[1][1] * jacobianGradient.y),
            0.f, 0.f);
    }
}
/* *************************************************************** */
__global__ void reg_spline_computeApproxJacGradient3D_kernel(float4 *gradient,
                                                             cudaTextureObject_t jacobianDeterminantTexture,
                                                             cudaTextureObject_t jacobianMatricesTexture,
                                                             const int3 controlPointImageDim,
                                                             const unsigned controlPointNumber,
                                                             const mat33 reorientation,
                                                             const float3 weight) {
    __shared__ float xbasis[27];
    __shared__ float ybasis[27];
    __shared__ float zbasis[27];

    if (threadIdx.x < 27)
        GetFirstDerivativeBasisValues3D(threadIdx.x, xbasis, ybasis, zbasis);
    __syncthreads();

    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float3 jacobianGradient{};
        unsigned index = 26;
        for (int pixelZ = z - 1; pixelZ < z + 2; ++pixelZ) {
            if (0 < pixelZ && pixelZ < controlPointImageDim.z - 1) {
                for (int pixelY = y - 1; pixelY < y + 2; ++pixelY) {
                    if (0 < pixelY && pixelY < controlPointImageDim.y - 1) {
                        int jacIndex = (pixelZ * controlPointImageDim.y + pixelY) * controlPointImageDim.x + x - 1;
                        for (int pixelX = x - 1; pixelX < x + 2; ++pixelX) {
                            if (0 < pixelX && pixelX < controlPointImageDim.x - 1) {
                                float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);
                                if (detJac > 0.f) {
                                    detJac = 2.f * logf(detJac) / detJac;
                                    float jacobianMatrix[9];
                                    jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9);
                                    jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 1);
                                    jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 2);
                                    jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 3);
                                    jacobianMatrix[4] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 4);
                                    jacobianMatrix[5] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 5);
                                    jacobianMatrix[6] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 6);
                                    jacobianMatrix[7] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 7);
                                    jacobianMatrix[8] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex * 9 + 8);
                                    GetJacobianGradientValues3D(jacobianMatrix, detJac, xbasis[index], ybasis[index], zbasis[index], &jacobianGradient);
                                }
                            }
                            jacIndex++;
                            index--;
                        }
                    } else index -= 3;
                }
            } else index -= 9;
        }

        gradient[tid] = gradient[tid] + make_float4(
            weight.x * (reorientation.m[0][0] * jacobianGradient.x + reorientation.m[0][1] * jacobianGradient.y + reorientation.m[0][2] * jacobianGradient.z),
            weight.y * (reorientation.m[1][0] * jacobianGradient.x + reorientation.m[1][1] * jacobianGradient.y + reorientation.m[1][2] * jacobianGradient.z),
            weight.z * (reorientation.m[2][0] * jacobianGradient.x + reorientation.m[2][1] * jacobianGradient.y + reorientation.m[2][2] * jacobianGradient.z),
            0.f);
    }
}
/* *************************************************************** */
__global__ void reg_spline_computeJacGradient2D_kernel(float4 *gradient,
                                                       cudaTextureObject_t jacobianDeterminantTexture,
                                                       cudaTextureObject_t jacobianMatricesTexture,
                                                       const int3 controlPointImageDim,
                                                       const float3 controlPointVoxelSpacing,
                                                       const unsigned controlPointNumber,
                                                       const int3 referenceImageDim,
                                                       const mat33 reorientation,
                                                       const float3 weight) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float2 jacobianGradient{};
        for (int pixelY = Ceil((y - 3) * controlPointVoxelSpacing.y); pixelY <= Ceil((y + 1) * controlPointVoxelSpacing.y); ++pixelY) {
            if (-1 < pixelY && pixelY < referenceImageDim.y) {
                const int yPre = (int)((float)pixelY / controlPointVoxelSpacing.y);
                float basis = (float)pixelY / controlPointVoxelSpacing.y - (float)yPre;
                float yBasis, yFirst;
                GetBSplineBasisValue(basis, y - yPre, &yBasis, &yFirst);

                for (int pixelX = Ceil((x - 3) * controlPointVoxelSpacing.x); pixelX <= Ceil((x + 1) * controlPointVoxelSpacing.x); ++pixelX) {
                    if (-1 < pixelX && pixelX < referenceImageDim.x && (yFirst != 0.f || yBasis != 0.f)) {
                        const int xPre = (int)((float)pixelX / controlPointVoxelSpacing.x);
                        basis = (float)pixelX / controlPointVoxelSpacing.x - (float)xPre;
                        float xBasis, xFirst;
                        GetBSplineBasisValue(basis, x - xPre, &xBasis, &xFirst);

                        int jacIndex = pixelY * referenceImageDim.x + pixelX;
                        float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);

                        if (detJac > 0.f && (xFirst != 0.f || xBasis != 0.f)) {
                            detJac = 2.f * logf(detJac) / detJac;
                            float jacobianMatrix[4];
                            jacIndex *= 4;
                            jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                            jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                            jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                            jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex);
                            const float2 basisValues = { xFirst * yBasis, xBasis * yFirst };
                            GetJacobianGradientValues2D(jacobianMatrix, detJac, basisValues.x, basisValues.y, &jacobianGradient);
                        }
                    }
                }
            }
        }
        gradient[tid] = gradient[tid] + make_float4(
            weight.x * (reorientation.m[0][0] * jacobianGradient.x + reorientation.m[0][1] * jacobianGradient.y),
            weight.y * (reorientation.m[1][0] * jacobianGradient.x + reorientation.m[1][1] * jacobianGradient.y),
            0.f, 0.f);
    }
}
/* *************************************************************** */
__global__ void reg_spline_computeJacGradient3D_kernel(float4 *gradient,
                                                       cudaTextureObject_t jacobianDeterminantTexture,
                                                       cudaTextureObject_t jacobianMatricesTexture,
                                                       const int3 controlPointImageDim,
                                                       const float3 controlPointVoxelSpacing,
                                                       const unsigned controlPointNumber,
                                                       const int3 referenceImageDim,
                                                       const mat33 reorientation,
                                                       const float3 weight) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float3 jacobianGradient{};
        for (int pixelZ = Ceil((z - 3) * controlPointVoxelSpacing.z); pixelZ <= Ceil((z + 1) * controlPointVoxelSpacing.z); ++pixelZ) {
            if (-1 < pixelZ && pixelZ < referenceImageDim.z) {
                const int zPre = (int)((float)pixelZ / controlPointVoxelSpacing.z);
                float basis = (float)pixelZ / controlPointVoxelSpacing.z - (float)zPre;
                float zBasis, zFirst;
                GetBSplineBasisValue(basis, z - zPre, &zBasis, &zFirst);

                for (int pixelY = Ceil((y - 3) * controlPointVoxelSpacing.y); pixelY <= Ceil((y + 1) * controlPointVoxelSpacing.y); ++pixelY) {
                    if (-1 < pixelY && pixelY < referenceImageDim.y && (zFirst != 0.f || zBasis != 0.f)) {
                        const int yPre = (int)((float)pixelY / controlPointVoxelSpacing.y);
                        basis = (float)pixelY / controlPointVoxelSpacing.y - (float)yPre;
                        float yBasis, yFirst;
                        GetBSplineBasisValue(basis, y - yPre, &yBasis, &yFirst);

                        for (int pixelX = Ceil((x - 3) * controlPointVoxelSpacing.x); pixelX <= Ceil((x + 1) * controlPointVoxelSpacing.x); ++pixelX) {
                            if (-1 < pixelX && pixelX < referenceImageDim.x && (yFirst != 0.f || yBasis != 0.f)) {
                                const int xPre = (int)((float)pixelX / controlPointVoxelSpacing.x);
                                basis = (float)pixelX / controlPointVoxelSpacing.x - (float)xPre;
                                float xBasis, xFirst;
                                GetBSplineBasisValue(basis, x - xPre, &xBasis, &xFirst);

                                int jacIndex = (pixelZ * referenceImageDim.y + pixelY) * referenceImageDim.x + pixelX;
                                float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);

                                if (detJac > 0.f && (xFirst != 0.f || xBasis != 0.f)) {
                                    detJac = 2.f * logf(detJac) / detJac;
                                    float jacobianMatrix[9];
                                    jacIndex *= 9;
                                    jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[4] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[5] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[6] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[7] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[8] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex);

                                    const float3 basisValues = {
                                        xFirst * yBasis * zBasis,
                                        xBasis * yFirst * zBasis,
                                        xBasis * yBasis * zFirst
                                    };
                                    GetJacobianGradientValues3D(jacobianMatrix, detJac, basisValues.x, basisValues.y, basisValues.z, &jacobianGradient);
                                }
                            }
                        }
                    }
                }
            }
        }
        gradient[tid] = gradient[tid] + make_float4(
            weight.x * (reorientation.m[0][0] * jacobianGradient.x + reorientation.m[0][1] * jacobianGradient.y + reorientation.m[0][2] * jacobianGradient.z),
            weight.y * (reorientation.m[1][0] * jacobianGradient.x + reorientation.m[1][1] * jacobianGradient.y + reorientation.m[1][2] * jacobianGradient.z),
            weight.z * (reorientation.m[2][0] * jacobianGradient.x + reorientation.m[2][1] * jacobianGradient.y + reorientation.m[2][2] * jacobianGradient.z),
            0.f);
    }
}
/* *************************************************************** */
__global__ void reg_spline_approxCorrectFolding3D_kernel(float4 *controlPointGrid,
                                                         cudaTextureObject_t jacobianDeterminantTexture,
                                                         cudaTextureObject_t jacobianMatricesTexture,
                                                         const int3 controlPointImageDim,
                                                         const float3 controlPointSpacing,
                                                         const unsigned controlPointNumber,
                                                         const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float3 foldingCorrection{};
        for (int pixelZ = z - 1; pixelZ < z + 2; ++pixelZ) {
            if (0 < pixelZ && pixelZ < controlPointImageDim.z - 1) {
                for (int pixelY = y - 1; pixelY < y + 2; ++pixelY) {
                    if (0 < pixelY && pixelY < controlPointImageDim.y - 1) {
                        for (int pixelX = x - 1; pixelX < x + 2; ++pixelX) {
                            if (0 < pixelX && pixelX < controlPointImageDim.x - 1) {
                                int jacIndex = (pixelZ * controlPointImageDim.y + pixelY) * controlPointImageDim.x + pixelX;
                                float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);
                                if (detJac <= 0.f) {
                                    float jacobianMatrix[9];
                                    jacIndex *= 9;
                                    jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[4] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[5] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[6] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[7] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[8] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex);

                                    float xBasis, xFirst, yBasis, yFirst, zBasis, zFirst;
                                    GetBSplineBasisValue(0.f, x - pixelX + 1, &xBasis, &xFirst);
                                    GetBSplineBasisValue(0.f, y - pixelY + 1, &yBasis, &yFirst);
                                    GetBSplineBasisValue(0.f, z - pixelZ + 1, &zBasis, &zFirst);

                                    const float3 basisValue = {
                                        xFirst * yBasis * zBasis,
                                        xBasis * yFirst * zBasis,
                                        xBasis * yBasis * zFirst
                                    };
                                    GetJacobianGradientValues3D(jacobianMatrix, 1.f, basisValue.x, basisValue.y, basisValue.z, &foldingCorrection);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (foldingCorrection.x != 0.f && foldingCorrection.y != 0.f && foldingCorrection.z != 0.f) {
            const float3 gradient = {
                reorientation.m[0][0] * foldingCorrection.x + reorientation.m[0][1] * foldingCorrection.y + reorientation.m[0][2] * foldingCorrection.z,
                reorientation.m[1][0] * foldingCorrection.x + reorientation.m[1][1] * foldingCorrection.y + reorientation.m[1][2] * foldingCorrection.z,
                reorientation.m[2][0] * foldingCorrection.x + reorientation.m[2][1] * foldingCorrection.y + reorientation.m[2][2] * foldingCorrection.z
            };
            const float norm = 5 * sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
            controlPointGrid[tid] = controlPointGrid[tid] + make_float4(gradient.x * controlPointSpacing.x / norm,
                                                                        gradient.y * controlPointSpacing.y / norm,
                                                                        gradient.z * controlPointSpacing.z / norm, 0.f);
        }
    }
}
/* *************************************************************** */
__global__ void reg_spline_correctFolding3D_kernel(float4 *controlPointGrid,
                                                   cudaTextureObject_t jacobianDeterminantTexture,
                                                   cudaTextureObject_t jacobianMatricesTexture,
                                                   const int3 controlPointImageDim,
                                                   const float3 controlPointSpacing,
                                                   const float3 controlPointVoxelSpacing,
                                                   const unsigned controlPointNumber,
                                                   const int3 referenceImageDim,
                                                   const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < controlPointNumber) {
        int quot, rem;
        reg_div_cuda(tid, controlPointImageDim.x * controlPointImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, controlPointImageDim.x, quot, rem);
        const int y = quot, x = rem;

        float3 foldingCorrection{};
        for (int pixelZ = Ceil((z - 3) * controlPointVoxelSpacing.z); pixelZ < Ceil((z + 1) * controlPointVoxelSpacing.z); ++pixelZ) {
            if (-1 < pixelZ && pixelZ < referenceImageDim.z) {
                for (int pixelY = Ceil((y - 3) * controlPointVoxelSpacing.y); pixelY < Ceil((y + 1) * controlPointVoxelSpacing.y); ++pixelY) {
                    if (-1 < pixelY && pixelY < referenceImageDim.y) {
                        for (int pixelX = Ceil((x - 3) * controlPointVoxelSpacing.x); pixelX < Ceil((x + 1) * controlPointVoxelSpacing.x); ++pixelX) {
                            if (-1 < pixelX && pixelX < referenceImageDim.x) {
                                int jacIndex = (pixelZ * referenceImageDim.y + pixelY) * referenceImageDim.x + pixelX;
                                float detJac = tex1Dfetch<float>(jacobianDeterminantTexture, jacIndex);
                                if (detJac <= 0.f) {
                                    float jacobianMatrix[9];
                                    jacIndex *= 9;
                                    jacobianMatrix[0] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[1] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[2] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[3] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[4] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[5] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[6] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[7] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex++);
                                    jacobianMatrix[8] = tex1Dfetch<float>(jacobianMatricesTexture, jacIndex);

                                    float xBasis, xFirst, yBasis, yFirst, zBasis, zFirst;
                                    int pre = (int)((float)pixelX / controlPointVoxelSpacing.x);
                                    float basis = (float)pixelX / controlPointVoxelSpacing.x - (float)pre;
                                    GetBSplineBasisValue(basis, x - pre, &xBasis, &xFirst);
                                    pre = (int)((float)pixelY / controlPointVoxelSpacing.y);
                                    basis = (float)pixelY / controlPointVoxelSpacing.y - (float)pre;
                                    GetBSplineBasisValue(basis, y - pre, &yBasis, &yFirst);
                                    pre = (int)((float)pixelZ / controlPointVoxelSpacing.z);
                                    basis = (float)pixelZ / controlPointVoxelSpacing.z - (float)pre;
                                    GetBSplineBasisValue(basis, z - pre, &zBasis, &zFirst);

                                    const float3 basisValue = {
                                        xFirst * yBasis * zBasis,
                                        xBasis * yFirst * zBasis,
                                        xBasis * yBasis * zFirst
                                    };
                                    GetJacobianGradientValues3D(jacobianMatrix, 1.f, basisValue.x, basisValue.y, basisValue.z, &foldingCorrection);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (foldingCorrection.x != 0.f && foldingCorrection.y != 0.f && foldingCorrection.z != 0.f) {
            const float3 gradient = {
                reorientation.m[0][0] * foldingCorrection.x + reorientation.m[0][1] * foldingCorrection.y + reorientation.m[0][2] * foldingCorrection.z,
                reorientation.m[1][0] * foldingCorrection.x + reorientation.m[1][1] * foldingCorrection.y + reorientation.m[1][2] * foldingCorrection.z,
                reorientation.m[2][0] * foldingCorrection.x + reorientation.m[2][1] * foldingCorrection.y + reorientation.m[2][2] * foldingCorrection.z
            };
            const float norm = 5.f * sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
            controlPointGrid[tid] = controlPointGrid[tid] + make_float4(gradient.x * controlPointSpacing.x / norm,
                                                                        gradient.y * controlPointSpacing.y / norm,
                                                                        gradient.z * controlPointSpacing.z / norm, 0.f);
        }
    }
}
/* *************************************************************** */
__global__ void reg_getDeformationFromDisplacement3D_kernel(float4 *image,
                                                            const int3 imageDim,
                                                            const unsigned voxelNumber,
                                                            const mat44 affineMatrix,
                                                            const bool reverse = false) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, imageDim.x * imageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, imageDim.x, quot, rem);
        const int y = quot, x = rem;

        const float4 initialPosition = {
            x * affineMatrix.m[0][0] + y * affineMatrix.m[0][1] + z * affineMatrix.m[0][2] + affineMatrix.m[0][3],
            x * affineMatrix.m[1][0] + y * affineMatrix.m[1][1] + z * affineMatrix.m[1][2] + affineMatrix.m[1][3],
            x * affineMatrix.m[2][0] + y * affineMatrix.m[2][1] + z * affineMatrix.m[2][2] + affineMatrix.m[2][3],
            0.f
        };

        // If reverse, gets displacement from deformation
        image[tid] = image[tid] + (reverse ? -1 : 1) * initialPosition;
    }
}
/* *************************************************************** */
__global__ void reg_defField_compose2D_kernel(float4 *deformationField,
                                              cudaTextureObject_t deformationFieldTexture,
                                              const int3 referenceImageDim,
                                              const unsigned voxelNumber,
                                              const mat44 affineMatrixB,
                                              const mat44 affineMatrixC) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        // Extract the original voxel position
        float4 position = deformationField[tid];

        // Conversion from real position to voxel coordinate
        float4 voxelPosition = {
            position.x * affineMatrixB.m[0][0] + position.y * affineMatrixB.m[0][1] + affineMatrixB.m[0][3],
            position.x * affineMatrixB.m[1][0] + position.y * affineMatrixB.m[1][1] + affineMatrixB.m[1][3],
            0.f,
            0.f
        };

        // Linear interpolation
        const int2 ante = { Floor(voxelPosition.x), Floor(voxelPosition.y) };
        float relX[2], relY[2];
        relX[1] = voxelPosition.x - (float)ante.x; relX[0] = 1.f - relX[1];
        relY[1] = voxelPosition.y - (float)ante.y; relY[0] = 1.f - relY[1];

        position = make_float4(0.f, 0.f, 0.f, 0.f);
        for (short b = 0; b < 2; ++b) {
            for (short a = 0; a < 2; ++a) {
                float4 deformation;
                if (-1 < ante.x + a && ante.x + a < referenceImageDim.x &&
                    -1 < ante.y + b && ante.y + b < referenceImageDim.y) {
                    const int index = (ante.y + b) * referenceImageDim.x + ante.x + a;
                    deformation = tex1Dfetch<float4>(deformationFieldTexture, index);
                } else {
                    deformation = GetSlidedValues(ante.x + a, ante.y + b, deformationFieldTexture, referenceImageDim, affineMatrixC);
                }
                const float basis = relX[a] * relY[b];
                position = position + basis * deformation;
            }
        }
        deformationField[tid] = position;
    }
}
/* *************************************************************** */
__global__ void reg_defField_compose3D_kernel(float4 *deformationField,
                                              cudaTextureObject_t deformationFieldTexture,
                                              const int3 referenceImageDim,
                                              const unsigned voxelNumber,
                                              const mat44 affineMatrixB,
                                              const mat44 affineMatrixC) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        // Extract the original voxel position
        float4 position = deformationField[tid];

        // Conversion from real position to voxel coordinate
        const float4 voxelPosition = {
            position.x * affineMatrixB.m[0][0] + position.y * affineMatrixB.m[0][1] + position.z * affineMatrixB.m[0][2] + affineMatrixB.m[0][3],
            position.x * affineMatrixB.m[1][0] + position.y * affineMatrixB.m[1][1] + position.z * affineMatrixB.m[1][2] + affineMatrixB.m[1][3],
            position.x * affineMatrixB.m[2][0] + position.y * affineMatrixB.m[2][1] + position.z * affineMatrixB.m[2][2] + affineMatrixB.m[2][3],
            0.f
        };

        // Linear interpolation
        const int3 ante = { Floor(voxelPosition.x), Floor(voxelPosition.y), Floor(voxelPosition.z) };
        float relX[2], relY[2], relZ[2];
        relX[1] = voxelPosition.x - (float)ante.x; relX[0] = 1.f - relX[1];
        relY[1] = voxelPosition.y - (float)ante.y; relY[0] = 1.f - relY[1];
        relZ[1] = voxelPosition.z - (float)ante.z; relZ[0] = 1.f - relZ[1];

        position = make_float4(0.f, 0.f, 0.f, 0.f);
        for (short c = 0; c < 2; ++c) {
            for (short b = 0; b < 2; ++b) {
                for (short a = 0; a < 2; ++a) {
                    float4 deformation;
                    if (-1 < ante.x + a && ante.x + a < referenceImageDim.x &&
                        -1 < ante.y + b && ante.y + b < referenceImageDim.y &&
                        -1 < ante.z + c && ante.z + c < referenceImageDim.z) {
                        const int index = ((ante.z + c) * referenceImageDim.y + ante.y + b) * referenceImageDim.x + ante.x + a;
                        deformation = tex1Dfetch<float4>(deformationFieldTexture, index);
                    } else {
                        deformation = GetSlidedValues(ante.x + a, ante.y + b, ante.z + c, deformationFieldTexture, referenceImageDim, affineMatrixC);
                    }
                    const float basis = relX[a] * relY[b] * relZ[c];
                    position = position + basis * deformation;
                }
            }
        }
        deformationField[tid] = position;
    }
}
/* *************************************************************** */
__global__ void reg_defField_getJacobianMatrix3D_kernel(float *jacobianMatrices,
                                                        cudaTextureObject_t deformationFieldTexture,
                                                        const int3 referenceImageDim,
                                                        const unsigned voxelNumber,
                                                        const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, referenceImageDim.x * referenceImageDim.y, quot, rem);
        const int z = quot;
        reg_div_cuda(rem, referenceImageDim.x, quot, rem);
        const int y = quot, x = rem;

        if (x == referenceImageDim.x - 1 || y == referenceImageDim.y - 1 || z == referenceImageDim.z - 1) {
            int index = tid * 9;
            jacobianMatrices[index++] = 1;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index++] = 1;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index++] = 0;
            jacobianMatrices[index] = 1;
            return;
        }

        int index = (z * referenceImageDim.y + y) * referenceImageDim.x + x;
        float4 deformation = tex1Dfetch<float4>(deformationFieldTexture, index);
        float matrix[9] = {
            -deformation.x, -deformation.x, -deformation.x,
            -deformation.y, -deformation.y, -deformation.y,
            -deformation.z, -deformation.z, -deformation.z
        };
        deformation = tex1Dfetch<float4>(deformationFieldTexture, index + 1);
        matrix[0] += deformation.x;
        matrix[3] += deformation.y;
        matrix[6] += deformation.z;
        index = (z * referenceImageDim.y + y + 1) * referenceImageDim.x + x;
        deformation = tex1Dfetch<float4>(deformationFieldTexture, index);
        matrix[1] += deformation.x;
        matrix[4] += deformation.y;
        matrix[7] += deformation.z;
        index = ((z + 1) * referenceImageDim.y + y) * referenceImageDim.x + x;
        deformation = tex1Dfetch<float4>(deformationFieldTexture, index);
        matrix[2] += deformation.x;
        matrix[5] += deformation.y;
        matrix[8] += deformation.z;

        index = tid * 9;
        jacobianMatrices[index++] = reorientation.m[0][0] * matrix[0] + reorientation.m[0][1] * matrix[3] + reorientation.m[0][2] * matrix[6];
        jacobianMatrices[index++] = reorientation.m[0][0] * matrix[1] + reorientation.m[0][1] * matrix[4] + reorientation.m[0][2] * matrix[7];
        jacobianMatrices[index++] = reorientation.m[0][0] * matrix[2] + reorientation.m[0][1] * matrix[5] + reorientation.m[0][2] * matrix[8];
        jacobianMatrices[index++] = reorientation.m[1][0] * matrix[0] + reorientation.m[1][1] * matrix[3] + reorientation.m[1][2] * matrix[6];
        jacobianMatrices[index++] = reorientation.m[1][0] * matrix[1] + reorientation.m[1][1] * matrix[4] + reorientation.m[1][2] * matrix[7];
        jacobianMatrices[index++] = reorientation.m[1][0] * matrix[2] + reorientation.m[1][1] * matrix[5] + reorientation.m[1][2] * matrix[8];
        jacobianMatrices[index++] = reorientation.m[2][0] * matrix[0] + reorientation.m[2][1] * matrix[3] + reorientation.m[2][2] * matrix[6];
        jacobianMatrices[index++] = reorientation.m[2][0] * matrix[1] + reorientation.m[2][1] * matrix[4] + reorientation.m[2][2] * matrix[7];
        jacobianMatrices[index] = reorientation.m[2][0] * matrix[2] + reorientation.m[2][1] * matrix[5] + reorientation.m[2][2] * matrix[8];
    }
}
/* *************************************************************** */
struct Basis {
    float x[27], y[27], z[27];
};
/* *************************************************************** */
template<bool is3d>
__device__ static mat33 CreateDisplacementMatrix(const unsigned index,
                                                 cudaTextureObject_t controlPointGridTexture,
                                                 const int3& cppDims,
                                                 const Basis& basis,
                                                 const mat33& reorientation) {
    const auto&& [x, y, z] = reg_indexToDims_cuda<is3d>((int)index, cppDims);
    if (x < 1 || x >= cppDims.x - 1 || y < 1 || y >= cppDims.y - 1 ||
        (is3d && (z < 1 || z >= cppDims.z - 1))) return {};

    mat33 matrix{};
    if constexpr (is3d) {
        for (int c = -1, basInd = 0; c < 2; c++) {
            const int zInd = (z + c) * cppDims.y;
            for (int b = -1; b < 2; b++) {
                const int yInd = (zInd + y + b) * cppDims.x;
                for (int a = -1; a < 2; a++, basInd++) {
                    const int index = yInd + x + a;
                    const float4 splineCoeff = tex1Dfetch<float4>(controlPointGridTexture, index);

                    matrix.m[0][0] += basis.x[basInd] * splineCoeff.x;
                    matrix.m[1][0] += basis.y[basInd] * splineCoeff.x;
                    matrix.m[2][0] += basis.z[basInd] * splineCoeff.x;

                    matrix.m[0][1] += basis.x[basInd] * splineCoeff.y;
                    matrix.m[1][1] += basis.y[basInd] * splineCoeff.y;
                    matrix.m[2][1] += basis.z[basInd] * splineCoeff.y;

                    matrix.m[0][2] += basis.x[basInd] * splineCoeff.z;
                    matrix.m[1][2] += basis.y[basInd] * splineCoeff.z;
                    matrix.m[2][2] += basis.z[basInd] * splineCoeff.z;
                }
            }
        }
    } else {
        matrix.m[2][2] = 1;
        for (int b = -1, basInd = 0; b < 2; b++) {
            const int yInd = (y + b) * cppDims.x;
            for (int a = -1; a < 2; a++, basInd++) {
                const int index = yInd + x + a;
                const float4 splineCoeff = tex1Dfetch<float4>(controlPointGridTexture, index);

                matrix.m[0][0] += basis.x[basInd] * splineCoeff.x;
                matrix.m[1][0] += basis.y[basInd] * splineCoeff.x;

                matrix.m[0][1] += basis.x[basInd] * splineCoeff.y;
                matrix.m[1][1] += basis.y[basInd] * splineCoeff.y;
            }
        }
    }
    // Convert from mm to voxel
    matrix = reg_mat33_mul_cuda(reorientation, matrix);
    // Removing the rotation component
    const mat33 r = reg_mat33_inverse_cuda(reg_mat33_polar_cuda(matrix));
    matrix = reg_mat33_mul_cuda(r, matrix);
    // Convert to displacement
    matrix.m[0][0]--; matrix.m[1][1]--;
    if constexpr (is3d) matrix.m[2][2]--;
    return matrix;
}
/* *************************************************************** */
template<bool is3d>
__global__ void reg_spline_createDisplacementMatrices_kernel(mat33 *dispMatrices,
                                                             cudaTextureObject_t controlPointGridTexture,
                                                             const int3 cppDims,
                                                             const Basis basis,
                                                             const mat33 reorientation,
                                                             const unsigned voxelNumber) {
    const unsigned index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < voxelNumber)
        dispMatrices[index] = CreateDisplacementMatrix<is3d>(index, controlPointGridTexture, cppDims, basis, reorientation);
}
/* *************************************************************** */
template<bool is3d>
__global__ void reg_spline_approxLinearEnergyGradient_kernel(float4 *transGradient,
                                                             cudaTextureObject_t dispMatricesTexture,
                                                             const int3 cppDims,
                                                             const float approxRatio,
                                                             const Basis basis,
                                                             const mat33 invReorientation,
                                                             const unsigned voxelNumber) {
    const unsigned index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= voxelNumber) return;
    const auto&& [x, y, z] = reg_indexToDims_cuda<is3d>((int)index, cppDims);
    auto gradVal = transGradient[index];

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

    transGradient[index] = gradVal;
}
/* *************************************************************** */
