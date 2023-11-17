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
template<typename T>
__inline__ __device__ void InterpLinearKernel(T relative, T (&basis)[2]) {
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
    if (tid >= activeVoxelNumber) return;
    // Get the real world deformation in the floating space
    const int tid2 = tex1Dfetch<int>(maskTexture, tid);
    const float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

    // Get the voxel-based deformation in the floating space
    double2 voxelDeformation;
    voxelDeformation.x = (double(floatingMatrix.m[0][0]) * double(realDeformation.x) +
                          double(floatingMatrix.m[0][1]) * double(realDeformation.y) +
                          double(floatingMatrix.m[0][3]));
    voxelDeformation.y = (double(floatingMatrix.m[1][0]) * double(realDeformation.x) +
                          double(floatingMatrix.m[1][1]) * double(realDeformation.y) +
                          double(floatingMatrix.m[1][3]));

    // Compute the linear interpolation
    const int2 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y) };
    const double2 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y };
    double xBasis[2], yBasis[2];
    InterpLinearKernel(relative.x, xBasis);
    InterpLinearKernel(relative.y, yBasis);

    double intensity = 0;
    int indexY = previous.y * floatingDim.x + previous.x;
    for (char b = 0; b < 2; b++, indexY += floatingDim.x) {
        const int y = previous.y + b;
        int index = indexY;
        double xTempNewValue = 0;
        for (char a = 0; a < 2; a++, index++) {
            const int x = previous.x + a;
            if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y) {
                xTempNewValue += tex1Dfetch<float>(floatingTexture, index) * xBasis[a];
            } else {
                // Padding value
                xTempNewValue += paddingValue * xBasis[a];
            }
        }
        intensity += xTempNewValue * yBasis[b];
    }

    resultArray[tid2] = intensity;
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
    if (tid >= activeVoxelNumber) return;
    // Get the real world deformation in the floating space
    const int tid2 = tex1Dfetch<int>(maskTexture, tid);
    const float4 realDeformation = tex1Dfetch<float4>(deformationFieldTexture, tid);

    // Get the voxel-based deformation in the floating space
    double3 voxelDeformation;
    voxelDeformation.x = (double(floatingMatrix.m[0][0]) * double(realDeformation.x) +
                          double(floatingMatrix.m[0][1]) * double(realDeformation.y) +
                          double(floatingMatrix.m[0][2]) * double(realDeformation.z) +
                          double(floatingMatrix.m[0][3]));
    voxelDeformation.y = (double(floatingMatrix.m[1][0]) * double(realDeformation.x) +
                          double(floatingMatrix.m[1][1]) * double(realDeformation.y) +
                          double(floatingMatrix.m[1][2]) * double(realDeformation.z) +
                          double(floatingMatrix.m[1][3]));
    voxelDeformation.z = (double(floatingMatrix.m[2][0]) * double(realDeformation.x) +
                          double(floatingMatrix.m[2][1]) * double(realDeformation.y) +
                          double(floatingMatrix.m[2][2]) * double(realDeformation.z) +
                          double(floatingMatrix.m[2][3]));

    // Compute the linear interpolation
    const int3 previous = { Floor(voxelDeformation.x), Floor(voxelDeformation.y), Floor(voxelDeformation.z) };
    const double3 relative = { voxelDeformation.x - previous.x, voxelDeformation.y - previous.y, voxelDeformation.z - previous.z };
    double xBasis[2], yBasis[2], zBasis[2];
    InterpLinearKernel(relative.x, xBasis);
    InterpLinearKernel(relative.y, yBasis);
    InterpLinearKernel(relative.z, zBasis);

    double intensity = 0;
    for (char c = 0; c < 2; c++) {
        const int z = previous.z + c;
        int indexYZ = (z * floatingDim.y + previous.y) * floatingDim.x;
        double yTempNewValue = 0;
        for (char b = 0; b < 2; b++, indexYZ += floatingDim.x) {
            const int y = previous.y + b;
            int index = indexYZ + previous.x;
            double xTempNewValue = 0;
            for (char a = 0; a < 2; a++, index++) {
                const int x = previous.x + a;
                if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z) {
                    xTempNewValue += tex1Dfetch<float>(floatingTexture, index) * xBasis[a];
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
/* *************************************************************** */
__global__ void reg_getImageGradient2D_kernel(float4 *gradientArray,
                                              cudaTextureObject_t floatingTexture,
                                              cudaTextureObject_t deformationFieldTexture,
                                              const mat44 floatingMatrix,
                                              const int3 floatingDim,
                                              const unsigned activeVoxelNumber,
                                              const float paddingValue) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= activeVoxelNumber) return;
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
    int indexY = previous.y * floatingDim.x + previous.x;
    for (char b = 0; b < 2; b++, indexY += floatingDim.x) {
        const int y = previous.y + b;
        int index = indexY;
        float2 tempValueX{};
        for (char a = 0; a < 2; a++, index++) {
            const int x = previous.x + a;
            float intensity = paddingValue;

            if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y)
                intensity = tex1Dfetch<float>(floatingTexture, index);

            tempValueX.x += intensity * deriv[a];
            tempValueX.y += intensity * xBasis[a];
        }
        gradientValue.x += tempValueX.x * yBasis[b];
        gradientValue.y += tempValueX.y * deriv[b];
    }

    gradientArray[tid] = gradientValue;
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
    if (tid >= activeVoxelNumber) return;
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
    for (char c = 0; c < 2; c++) {
        const int z = previous.z + c;
        int indexYZ = (z * floatingDim.y + previous.y) * floatingDim.x;
        float3 tempValueY{};
        for (char b = 0; b < 2; b++, indexYZ += floatingDim.x) {
            const int y = previous.y + b;
            int index = indexYZ + previous.x;
            float2 tempValueX{};
            for (char a = 0; a < 2; a++, index++) {
                const int x = previous.x + a;
                float intensity = paddingValue;

                if (-1 < x && x < floatingDim.x && -1 < y && y < floatingDim.y && -1 < z && z < floatingDim.z)
                    intensity = tex1Dfetch<float>(floatingTexture, index);

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
/* *************************************************************** */
