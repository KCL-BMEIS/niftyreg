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
__inline__ __device__ void InterpLinearKernel(float relative, float (&basis)[2]) {
    if (relative < 0)
        relative = 0;  // reg_rounding error
    basis[1] = relative;
    basis[0] = 1.f - relative;
}
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
        // Get the real world deformation in the floating space
        const int tid2 = tex1Dfetch<int>(maskTexture, tid);
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        // Get the voxel-based deformation in the floating space
        float2 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][3]);

        // Compute the linear interpolation
        const int2 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y) };
        const float2 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y };
        float xBasis[2], yBasis[2];
        InterpLinearKernel(relative.x, xBasis);
        InterpLinearKernel(relative.y, yBasis);

        float intensity = 0;
        for (short b = 0; b < 2; b++) {
            const int y = previous.y + b;
            float xTempNewValue = 0;
            for (short a = 0; a < 2; a++) {
                const int x = previous.x + a;
                if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y) {
                    xTempNewValue += tex3D<float>(floatingTexture, x, y, 0) * xBasis[a];
                } else {
                    // Padding value
                    xTempNewValue += paddingValue * xBasis[a];
                }
            }
            intensity += xTempNewValue * yBasis[b];
        }

        resultArray[tid2] = intensity;
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

        // Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        // Get the voxel-based deformation in the floating space
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

        // Compute the linear interpolation
        const int3 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y), Floor(voxelDeformation.z) };
        const float3 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y, voxelDeformation.z - previous.z };
        float xBasis[2], yBasis[2], zBasis[2];
        InterpLinearKernel(relative.x, xBasis);
        InterpLinearKernel(relative.y, yBasis);
        InterpLinearKernel(relative.z, zBasis);

        float intensity = 0;
        for (short c = 0; c < 2; c++) {
            const int z = previous.z + c;
            float yTempNewValue = 0;
            for (short b = 0; b < 2; b++) {
                const int y = previous.y + b;
                float xTempNewValue = 0;
                for (short a = 0; a < 2; a++) {
                    const int x = previous.x + a;
                    if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z) {
                        xTempNewValue += tex3D<float>(floatingTexture, x, y, z) * xBasis[a];
                    } else {
                        // Padding value
                        xTempNewValue += paddingValue * xBasis[a];
                    }
                }
                yTempNewValue += xTempNewValue * yBasis[b];
            }
            intensity += yTempNewValue * zBasis[c];
        }

        resultArray[tid2] = intensity;
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
        // Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        // Get the voxel-based deformation in the floating space
        float2 voxelDeformation;
        voxelDeformation.x = (floatingMatrix.m[0][0] * realDeformation.x +
                              floatingMatrix.m[0][1] * realDeformation.y +
                              floatingMatrix.m[0][3]);
        voxelDeformation.y = (floatingMatrix.m[1][0] * realDeformation.x +
                              floatingMatrix.m[1][1] * realDeformation.y +
                              floatingMatrix.m[1][3]);

        // Compute the gradient
        const int2 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y) };
        float xBasis[2], yBasis[2];
        const float2 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y };
        InterpLinearKernel(relative.x, xBasis);
        InterpLinearKernel(relative.y, yBasis);
        constexpr float deriv[] = { -1.0f, 1.0f };

        float4 gradientValue{};
        for (short b = 0; b < 2; b++) {
            float2 tempValueX{};
            const int y = previous.y + b;
            for (short a = 0; a < 2; a++) {
                const int x = previous.x + a;
                float intensity = paddingValue;

                if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y)
                    intensity = tex3D<float>(floatingTexture, x, y, 0);

                tempValueX.x += intensity * deriv[a];
                tempValueX.y += intensity * xBasis[a];
            }
            gradientValue.x += tempValueX.x * yBasis[b];
            gradientValue.y += tempValueX.y * deriv[b];
        }

        if (gradientValue.x != gradientValue.x)
            gradientValue.x = 0;
        if (gradientValue.y != gradientValue.y)
            gradientValue.y = 0;

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
        // Get the real world deformation in the floating space
        float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

        // Get the voxel-based deformation in the floating space
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

        // Compute the gradient
        const int3 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y), Floor(voxelDeformation.z) };
        float xBasis[2], yBasis[2], zBasis[2];
        const float3 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y, voxelDeformation.z - previous.z };
        InterpLinearKernel(relative.x, xBasis);
        InterpLinearKernel(relative.y, yBasis);
        InterpLinearKernel(relative.z, zBasis);
        constexpr float deriv[] = { -1.0f, 1.0f };

        float4 gradientValue{};
        for (short c = 0; c < 2; c++) {
            const int z = previous.z + c;
            float3 tempValueY{};
            for (short b = 0; b < 2; b++) {
                float2 tempValueX{};
                const int y = previous.y + b;
                for (short a = 0; a < 2; a++) {
                    const int x = previous.x + a;
                    float intensity = paddingValue;

                    if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z)
                        intensity = tex3D<float>(floatingTexture, x, y, z);

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

        if (gradientValue.x != gradientValue.x)
            gradientValue.x = 0;
        if (gradientValue.y != gradientValue.y)
            gradientValue.y = 0;
        if (gradientValue.z != gradientValue.z)
            gradientValue.z = 0;

        gradientArray[tid] = gradientValue;
    }
}
/* *************************************************************** */
