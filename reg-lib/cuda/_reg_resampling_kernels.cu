/*
 *  _reg_resampling_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

/* *************************************************************** */
__global__ void reg_resampleImage2D_kernel(float *resultArray,
                                           cudaTextureObject_t floatingTexture,
                                           cudaTextureObject_t deformationFieldTexture,
                                           cudaTextureObject_t maskTexture,
                                           const mat44 floatingMatrix,
                                           const int3 floatingDim,
                                           const unsigned activeVoxelNumber,
                                           const float paddingValue) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        //Get the real world deformation in the floating space
        const int tid2 = tex1Dfetch<int>(maskTexture, tid);
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        //Get the voxel-based deformation in the floating space
        float2 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][3]);

        if (voxelDeformation.x >= 0.0f && voxelDeformation.x <= floatingDim.x - 1 &&
            voxelDeformation.y >= 0.0f && voxelDeformation.y <= floatingDim.y - 1) {
            resultArray[tid2] = tex3D<float>(floatingTexture, voxelDeformation.x + 0.5f, voxelDeformation.y + 0.5f, 0.5f);
        } else resultArray[tid2] = paddingValue;
    }
}
/* *************************************************************** */
__global__ void reg_resampleImage3D_kernel(float *resultArray,
                                           cudaTextureObject_t floatingTexture,
                                           cudaTextureObject_t deformationFieldTexture,
                                           cudaTextureObject_t maskTexture,
                                           const mat44 floatingMatrix,
                                           const int3 floatingDim,
                                           const unsigned activeVoxelNumber,
                                           const float paddingValue) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        const int tid2 = tex1Dfetch<int>(maskTexture, tid);

        //Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        //Get the voxel-based deformation in the floating space
        float3 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][2] * realDeformation.z +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][2] * realDeformation.z +
                              floatingMatrix.m[1][3]);
        voxelDeformation.z = (floatingMatrix.m[2][0] * realDeformation.x +
                              floatingMatrix.m[2][1] * realDeformation.y +
                              floatingMatrix.m[2][2] * realDeformation.z +
                              floatingMatrix.m[2][3]);

        if (voxelDeformation.x >= 0.0f && voxelDeformation.x <= floatingDim.x - 1 &&
            voxelDeformation.y >= 0.0f && voxelDeformation.y <= floatingDim.y - 1 &&
            voxelDeformation.z >= 0.0f && voxelDeformation.z <= floatingDim.z - 1) {
            resultArray[tid2] = tex3D<float>(floatingTexture, voxelDeformation.x + 0.5f, voxelDeformation.y + 0.5f, voxelDeformation.z + 0.5f);
        } else resultArray[tid2] = paddingValue;
    }
}
/* *************************************************************** */
__global__ void reg_getImageGradient2D_kernel(float4 *gradientArray,
                                              cudaTextureObject_t floatingTexture,
                                              cudaTextureObject_t deformationFieldTexture,
                                              const mat44 floatingMatrix,
                                              const int3 floatingDim,
                                              const unsigned activeVoxelNumber,
                                              const float paddingValue) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        //Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        //Get the voxel-based deformation in the floating space
        float3 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][3]);

        int2 voxel;
        voxel.x = (int)(voxelDeformation.x);
        voxel.y = (int)(voxelDeformation.y);

        float xBasis[2];
        float relative = fabsf(voxelDeformation.x - (float)voxel.x);
        xBasis[0] = 1.0f - relative;
        xBasis[1] = relative;
        float yBasis[2];
        relative = fabsf(voxelDeformation.y - (float)voxel.y);
        yBasis[0] = 1.0f - relative;
        yBasis[1] = relative;
        float deriv[2];
        deriv[0] = -1.0f;
        deriv[1] = 1.0f;

        float4 gradientValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 relativeDeformation;
        for (short b = 0; b < 2; b++) {
            float2 tempValueX = make_float2(0.0f, 0.0f);
            relativeDeformation.y = ((float)voxel.y + (float)b + 0.5f) / (float)floatingDim.y;
            for (short a = 0; a < 2; a++) {
                relativeDeformation.x = ((float)voxel.x + (float)a + 0.5f) / (float)floatingDim.x;
                float intensity = paddingValue;

                if (0.f <= relativeDeformation.x && relativeDeformation.x <= 1.f &&
                    0.f <= relativeDeformation.y && relativeDeformation.y <= 1.f)
                    intensity = tex3D<float>(floatingTexture, relativeDeformation.x, relativeDeformation.y, 0.5f);

                tempValueX.x += intensity * deriv[a];
                tempValueX.y += intensity * xBasis[a];
            }
            gradientValue.x += tempValueX.x * yBasis[b];
            gradientValue.y += tempValueX.y * deriv[b];
        }
        gradientArray[tid] = gradientValue;
    }
}
/* *************************************************************** */
__global__ void reg_getImageGradient3D_kernel(float4 *gradientArray,
                                              cudaTextureObject_t floatingTexture,
                                              cudaTextureObject_t deformationFieldTexture,
                                              const mat44 floatingMatrix,
                                              const int3 floatingDim,
                                              const unsigned activeVoxelNumber,
                                              const float paddingValue) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < activeVoxelNumber) {
        //Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        //Get the voxel-based deformation in the floating space
        float3 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][2] * realDeformation.z +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][2] * realDeformation.z +
                              floatingMatrix.m[1][3]);
        voxelDeformation.z = (floatingMatrix.m[2][0] * realDeformation.x +
                              floatingMatrix.m[2][1] * realDeformation.y +
                              floatingMatrix.m[2][2] * realDeformation.z +
                              floatingMatrix.m[2][3]);

        int3 voxel;
        voxel.x = (int)(voxelDeformation.x);
        voxel.y = (int)(voxelDeformation.y);
        voxel.z = (int)(voxelDeformation.z);

        float xBasis[2];
        float relative = fabsf(voxelDeformation.x - (float)voxel.x);
        xBasis[0] = 1.0f - relative;
        xBasis[1] = relative;
        float yBasis[2];
        relative = fabsf(voxelDeformation.y - (float)voxel.y);
        yBasis[0] = 1.0f - relative;
        yBasis[1] = relative;
        float zBasis[2];
        relative = fabsf(voxelDeformation.z - (float)voxel.z);
        zBasis[0] = 1.0f - relative;
        zBasis[1] = relative;
        float deriv[2];
        deriv[0] = -1.0f;
        deriv[1] = 1.0f;

        float4 gradientValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 relativeDeformation;
        for (short c = 0; c < 2; c++) {
            relativeDeformation.z = ((float)voxel.z + (float)c + 0.5f) / (float)floatingDim.z;
            float3 tempValueY = make_float3(0.0f, 0.0f, 0.0f);
            for (short b = 0; b < 2; b++) {
                float2 tempValueX = make_float2(0.0f, 0.0f);
                relativeDeformation.y = ((float)voxel.y + (float)b + 0.5f) / (float)floatingDim.y;
                for (short a = 0; a < 2; a++) {
                    relativeDeformation.x = ((float)voxel.x + (float)a + 0.5f) / (float)floatingDim.x;
                    float intensity = paddingValue;

                    if (0.f <= relativeDeformation.x && relativeDeformation.x <= 1.f &&
                        0.f <= relativeDeformation.y && relativeDeformation.y <= 1.f &&
                        0.f <= relativeDeformation.z && relativeDeformation.z <= 1.f)
                        intensity = tex3D<float>(floatingTexture, relativeDeformation.x, relativeDeformation.y, relativeDeformation.z);

                    tempValueX.x += intensity * deriv[a];
                    tempValueX.y += intensity * xBasis[a];
                }
                tempValueY.x += tempValueX.x * yBasis[b];
                tempValueY.y += tempValueX.y * deriv[b];
                tempValueY.z += tempValueX.y * yBasis[b];
            }
            gradientValue.x += tempValueY.x * zBasis[c];
            gradientValue.y += tempValueY.y * zBasis[c];
            gradientValue.z += tempValueY.z * deriv[c];
        }
        gradientArray[tid] = gradientValue;
    }
}
/* *************************************************************** */
