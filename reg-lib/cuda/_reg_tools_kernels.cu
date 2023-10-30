/*
 *  _reg_tools_kernels.cu
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
template<bool is3d>
__global__ void reg_voxelCentricToNodeCentric_kernel(float4 *nodeImageCuda,
                                                     cudaTextureObject_t voxelImageTexture,
                                                     const unsigned nodeNumber,
                                                     const int3 nodeImageDims,
                                                     const int3 voxelImageDims,
                                                     const float weight,
                                                     const mat44 transformation,
                                                     const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= nodeNumber) return;
    // Calculate the node coordinates
    auto&& [x, y, z] = reg_indexToDims_cuda<is3d>(tid, nodeImageDims);
    // Transform into voxel coordinates
    float voxelCoord[3], nodeCoord[3] = { static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };
    reg_mat44_mul_cuda<is3d>(transformation, nodeCoord, voxelCoord);

    // Linear interpolation
    float basisX[2], basisY[2], basisZ[2], interpolatedValue[3]{};
    const int pre[3] = { Floor(voxelCoord[0]), Floor(voxelCoord[1]), Floor(voxelCoord[2]) };
    basisX[1] = voxelCoord[0] - static_cast<float>(pre[0]);
    basisX[0] = 1.f - basisX[1];
    basisY[1] = voxelCoord[1] - static_cast<float>(pre[1]);
    basisY[0] = 1.f - basisY[1];
    if constexpr (is3d) {
        basisZ[1] = voxelCoord[2] - static_cast<float>(pre[2]);
        basisZ[0] = 1.f - basisZ[1];
    }
    for (char c = 0; c < 2; c++) {
        const int indexZ = pre[2] + c;
        if (-1 < indexZ && indexZ < voxelImageDims.z) {
            for (char b = 0; b < 2; b++) {
                const int indexY = pre[1] + b;
                if (-1 < indexY && indexY < voxelImageDims.y) {
                    for (char a = 0; a < 2; a++) {
                        const int indexX = pre[0] + a;
                        if (-1 < indexX && indexX < voxelImageDims.x) {
                            const int index = (indexZ * voxelImageDims.y + indexY) * voxelImageDims.x + indexX;
                            float linearWeight = basisX[a] * basisY[b];
                            if constexpr (is3d) linearWeight *= basisZ[c];
                            const float4& voxelValue = tex1Dfetch<float4>(voxelImageTexture, index);
                            interpolatedValue[0] += linearWeight * voxelValue.x;
                            interpolatedValue[1] += linearWeight * voxelValue.y;
                            if constexpr (is3d)
                                interpolatedValue[2] += linearWeight * voxelValue.z;
                        }
                    }
                }
            }
        }
    }

    float reorientedValue[3];
    reg_mat33_mul_cuda<is3d>(reorientation, interpolatedValue, weight, reorientedValue);
    nodeImageCuda[tid] = { reorientedValue[0], reorientedValue[1], reorientedValue[2], 0 };
}
/* *************************************************************** */
__global__ void reg_convertNMIGradientFromVoxelToRealSpace_kernel(float4 *gradient, const mat44 matrix, const unsigned nodeNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nodeNumber) {
        const float4 voxelGradient = gradient[tid];
        float4 realGradient;
        realGradient.x = matrix.m[0][0] * voxelGradient.x + matrix.m[0][1] * voxelGradient.y + matrix.m[0][2] * voxelGradient.z;
        realGradient.y = matrix.m[1][0] * voxelGradient.x + matrix.m[1][1] * voxelGradient.y + matrix.m[1][2] * voxelGradient.z;
        realGradient.z = matrix.m[2][0] * voxelGradient.x + matrix.m[2][1] * voxelGradient.y + matrix.m[2][2] * voxelGradient.z;
        gradient[tid] = realGradient;
    }
}
/* *************************************************************** */
__global__ void reg_applyConvolutionWindowAlongX_kernel(float4 *smoothedImage,
                                                        cudaTextureObject_t imageTexture,
                                                        cudaTextureObject_t kernelTexture,
                                                        const int kernelSize,
                                                        const int3 imageSize,
                                                        const unsigned voxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, imageSize.x * imageSize.y, quot, rem);
        reg_div_cuda(rem, imageSize.x, quot, rem);
        int x = rem;

        const int radius = (kernelSize - 1) / 2;
        int index = tid - radius;
        x -= radius;

        float4 finalValue{};

        // Kahan summation used here
        float3 c{}, Y, t;
        float windowValue;
        for (int i = 0; i < kernelSize; i++) {
            if (-1 < x && x < imageSize.x) {
                float4 gradientValue = tex1Dfetch<float4>(imageTexture, index);
                windowValue = tex1Dfetch<float>(kernelTexture, i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index++;
            x++;
        }
        smoothedImage[tid] = finalValue;
    }
}
/* *************************************************************** */
__global__ void reg_applyConvolutionWindowAlongY_kernel(float4 *smoothedImage,
                                                        cudaTextureObject_t imageTexture,
                                                        cudaTextureObject_t kernelTexture,
                                                        const int kernelSize,
                                                        const int3 imageSize,
                                                        const unsigned voxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int quot, rem;
        reg_div_cuda(tid, imageSize.x * imageSize.y, quot, rem);
        int y = rem / imageSize.x;

        const int radius = (kernelSize - 1) / 2;
        int index = tid - imageSize.x * radius;
        y -= radius;

        float4 finalValue{};

        // Kahan summation used here
        float3 c{}, Y, t;
        float windowValue;
        for (int i = 0; i < kernelSize; i++) {
            if (-1 < y && y < imageSize.y) {
                float4 gradientValue = tex1Dfetch<float4>(imageTexture, index);
                windowValue = tex1Dfetch<float>(kernelTexture, i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index += imageSize.x;
            y++;
        }
        smoothedImage[tid] = finalValue;
    }
}
/* *************************************************************** */
__global__ void reg_applyConvolutionWindowAlongZ_kernel(float4 *smoothedImage,
                                                        cudaTextureObject_t imageTexture,
                                                        cudaTextureObject_t kernelTexture,
                                                        const int kernelSize,
                                                        const int3 imageSize,
                                                        const unsigned voxelNumber) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < voxelNumber) {
        int z = (int)tid / (imageSize.x * imageSize.y);

        const int radius = (kernelSize - 1) / 2;
        int index = tid - imageSize.x * imageSize.y * radius;
        z -= radius;

        float4 finalValue{};

        // Kahan summation used here
        float3 c{}, Y, t;
        float windowValue;
        for (int i = 0; i < kernelSize; i++) {
            if (-1 < z && z < imageSize.z) {
                float4 gradientValue = tex1Dfetch<float4>(imageTexture, index);
                windowValue = tex1Dfetch<float>(kernelTexture, i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index += imageSize.x * imageSize.y;
            z++;
        }
        smoothedImage[tid] = finalValue;
    }
}
/* *************************************************************** */
__global__ void reg_multiplyValue_kernel_float(float *array, const float value, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count)
        array[tid] *= value;
}
/* *************************************************************** */
__global__ void reg_multiplyValue_kernel_float4(float4 *array, const float value, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count) {
        const float4 temp = array[tid];
        array[tid] = make_float4(temp.x * value, temp.y * value, temp.z * value, temp.w * value);
    }
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float(float *array, const float value, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count)
        array[tid] += value;
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float4(float4 *array, const float value, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count) {
        const float4 temp = array[tid];
        array[tid] = make_float4(temp.x + value, temp.y + value, temp.z + value, temp.w + value);
    }
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float(float *array1, float *array2, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count)
        array1[tid] *= array2[tid];
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float4(float4 *array1, float4 *array2, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count) {
        const float4 a = array1[tid];
        const float4 b = array2[tid];
        array1[tid] = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float(float *array1, float *array2, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count)
        array1[tid] += array2[tid];
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float4(float4 *array1, float4 *array2, const unsigned count) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < count) {
        const float4 a = array1[tid];
        const float4 b = array2[tid];
        array1[tid] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}
/* *************************************************************** */
