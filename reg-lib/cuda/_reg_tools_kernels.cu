/*
 *  _reg_tools_kernels.cu
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

/* *************************************************************** */
__device__ __constant__ int c_NodeNumber;
__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int3 c_TargetImageDim;
__device__ __constant__ float3 c_VoxelNodeRatio;
__device__ __constant__ int3 c_ControlPointImageDim;
__device__ __constant__ int3 c_ImageDim;
__device__ __constant__ float c_Weight;
/* *************************************************************** */
texture<float4, 1, cudaReadModeElementType> controlPointTexture;
texture<float4, 1, cudaReadModeElementType> gradientImageTexture;
texture<float4, 1, cudaReadModeElementType> matrixTexture;
texture<float, 1, cudaReadModeElementType> convolutionKernelTexture;
/* *************************************************************** */
__device__ __inline__ void reg_mat33_mul_cuda(const mat33& mat, const float (&in)[3], const float& weight, float (&out)[3], const bool& is3d) {
    out[0] = weight * (mat.m[0][0] * in[0] + mat.m[0][1] * in[1] + mat.m[0][2] * in[2]);
    out[1] = weight * (mat.m[1][0] * in[0] + mat.m[1][1] * in[1] + mat.m[1][2] * in[2]);
    out[2] = is3d ? weight * (mat.m[2][0] * in[0] + mat.m[2][1] * in[1] + mat.m[2][2] * in[2]) : 0;
}
/* *************************************************************** */
__device__ __inline__ void reg_mat44_mul_cuda(const mat44& mat, const float (&in)[3], float (&out)[3], const bool& is3d) {
    out[0] = mat.m[0][0] * in[0] + mat.m[0][1] * in[1] + mat.m[0][2] * in[2] + mat.m[0][3];
    out[1] = mat.m[1][0] * in[0] + mat.m[1][1] * in[1] + mat.m[1][2] * in[2] + mat.m[1][3];
    out[2] = is3d ? mat.m[2][0] * in[0] + mat.m[2][1] * in[1] + mat.m[2][2] * in[2] + mat.m[2][3] : 0;
}
/* *************************************************************** */
__global__ void reg_voxelCentric2NodeCentric_kernel(float4 *nodeImageCuda,
                                                    cudaTextureObject_t voxelImageTexture,
                                                    const unsigned nodeNumber,
                                                    const int3 nodeImageDims,
                                                    const int3 voxelImageDims,
                                                    const bool is3d,
                                                    const float weight,
                                                    const mat44 transformation,
                                                    const mat33 reorientation) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nodeNumber) {
        float nodeCoord[3], voxelCoord[3], reorientedValue[3];
        int tempIndex = tid;
        nodeCoord[2] = tempIndex / (nodeImageDims.x * nodeImageDims.y);
        tempIndex -= nodeCoord[2] * nodeImageDims.x * nodeImageDims.y;
        nodeCoord[1] = tempIndex / nodeImageDims.x;
        nodeCoord[0] = tempIndex - nodeCoord[1] * nodeImageDims.x;
        reg_mat44_mul_cuda(transformation, nodeCoord, voxelCoord, is3d);

        // Linear interpolation
        float basisX[2], basisY[2], basisZ[2], interpolatedValue[3]{};
        const int pre[3] = { reg_floor(voxelCoord[0]), reg_floor(voxelCoord[1]), reg_floor(voxelCoord[2]) };
        basisX[1] = voxelCoord[0] - static_cast<float>(pre[0]);
        basisX[0] = 1.f - basisX[1];
        basisY[1] = voxelCoord[1] - static_cast<float>(pre[1]);
        basisY[0] = 1.f - basisY[1];
        if (is3d) {
            basisZ[1] = voxelCoord[2] - static_cast<float>(pre[2]);
            basisZ[0] = 1.f - basisZ[1];
        }
        for (short c = 0; c < 2; ++c) {
            const int indexZ = pre[2] + c;
            if (-1 < indexZ && indexZ < voxelImageDims.z) {
                for (short b = 0; b < 2; ++b) {
                    const int indexY = pre[1] + b;
                    if (-1 < indexY && indexY < voxelImageDims.y) {
                        for (short a = 0; a < 2; ++a) {
                            const int indexX = pre[0] + a;
                            if (-1 < indexX && indexX < voxelImageDims.x) {
                                const int index = (indexZ * voxelImageDims.y + indexY) * voxelImageDims.x + indexX;
                                const float linearWeight = basisX[a] * basisY[b] * (is3d ? basisZ[c] : 1);
                                const float4 voxelValue = tex1Dfetch<float4>(voxelImageTexture, index);
                                interpolatedValue[0] += linearWeight * voxelValue.x;
                                interpolatedValue[1] += linearWeight * voxelValue.y;
                                if (is3d)
                                    interpolatedValue[2] += linearWeight * voxelValue.z;
                            }
                        }
                    }
                }
            }
        }

        reg_mat33_mul_cuda(reorientation, interpolatedValue, weight, reorientedValue, is3d);
        nodeImageCuda[tid] = { reorientedValue[0], reorientedValue[1], reorientedValue[2], 0 };
    }
}
/* *************************************************************** */
__global__ void _reg_convertNMIGradientFromVoxelToRealSpace_kernel(float4 *gradient) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_NodeNumber) {
        float4 voxelGradient = gradient[tid];
        float4 realGradient;
        float4 matrix = tex1Dfetch(matrixTexture, 0);
        realGradient.x = matrix.x * voxelGradient.x + matrix.y * voxelGradient.y + matrix.z * voxelGradient.z;
        matrix = tex1Dfetch(matrixTexture, 1);
        realGradient.y = matrix.x * voxelGradient.x + matrix.y * voxelGradient.y + matrix.z * voxelGradient.z;
        matrix = tex1Dfetch(matrixTexture, 2);
        realGradient.z = matrix.x * voxelGradient.x + matrix.y * voxelGradient.y + matrix.z * voxelGradient.z;

        gradient[tid] = realGradient;
    }
}
/* *************************************************************** */
__global__ void _reg_ApplyConvolutionWindowAlongX_kernel(float4 *smoothedImage, int windowSize) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        int3 imageSize = c_ImageDim;

        int temp = tid;
        const short z = (int)(temp / (imageSize.x * imageSize.y));
        temp -= z * imageSize.x * imageSize.y;
        const short y = (int)(temp / (imageSize.x));
        short x = temp - y * (imageSize.x);

        int radius = (windowSize - 1) / 2;
        int index = tid - radius;
        x -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c = make_float3(0.f, 0.f, 0.f), Y, t;
        float windowValue;
        for (int i = 0; i < windowSize; i++) {
            if (-1 < x && x < imageSize.x) {
                float4 gradientValue = tex1Dfetch(gradientImageTexture, index);
                windowValue = tex1Dfetch(convolutionKernelTexture, i);

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
__global__ void _reg_ApplyConvolutionWindowAlongY_kernel(float4 *smoothedImage, int windowSize) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        int3 imageSize = c_ImageDim;

        const short z = (int)(tid / (imageSize.x * imageSize.y));
        int index = tid - z * imageSize.x * imageSize.y;
        short y = (int)(index / imageSize.x);

        int radius = (windowSize - 1) / 2;
        index = tid - imageSize.x * radius;
        y -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c = make_float3(0.f, 0.f, 0.f), Y, t;
        float windowValue;
        for (int i = 0; i < windowSize; i++) {
            if (-1 < y && y < imageSize.y) {
                float4 gradientValue = tex1Dfetch(gradientImageTexture, index);
                windowValue = tex1Dfetch(convolutionKernelTexture, i);

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
__global__ void _reg_ApplyConvolutionWindowAlongZ_kernel(float4 *smoothedImage, int windowSize) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        int3 imageSize = c_ImageDim;

        short z = (int)(tid / ((imageSize.x) * (imageSize.y)));

        int radius = (windowSize - 1) / 2;
        int index = tid - imageSize.x * imageSize.y * radius;
        z -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c = make_float3(0.f, 0.f, 0.f), Y, t;
        float windowValue;
        for (int i = 0; i < windowSize; i++) {
            if (-1 < z && z < imageSize.z) {
                float4 gradientValue = tex1Dfetch(gradientImageTexture, index);
                windowValue = tex1Dfetch(convolutionKernelTexture, i);

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
__global__ void reg_multiplyValue_kernel_float(float *array_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        array_d[tid] *= c_Weight;
    }
}
/* *************************************************************** */
__global__ void reg_multiplyValue_kernel_float4(float4 *array_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        float4 temp = array_d[tid];
        array_d[tid] = make_float4(temp.x * c_Weight, temp.y * c_Weight, temp.z * c_Weight, temp.w * c_Weight);
    }
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float(float *array_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        array_d[tid] += c_Weight;
    }
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float4(float4 *array_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        float4 temp = array_d[tid];
        array_d[tid] = make_float4(temp.x + c_Weight, temp.y + c_Weight, temp.z + c_Weight, temp.w + c_Weight);
    }
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float(float *array1_d, float *array2_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        array1_d[tid] *= array2_d[tid];
    }
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float4(float4 *array1_d, float4 *array2_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        float4 a = array1_d[tid];
        float4 b = array1_d[tid];
        array1_d[tid] = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float(float *array1_d, float *array2_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        array1_d[tid] += array2_d[tid];
    }
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float4(float4 *array1_d, float4 *array2_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber) {
        float4 a = array1_d[tid];
        float4 b = array1_d[tid];
        array1_d[tid] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}
/* *************************************************************** */
__global__ void reg_fillMaskArray_kernel(int *array1_d) {
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < c_VoxelNumber)
        array1_d[tid] = tid;
}
/* *************************************************************** */
