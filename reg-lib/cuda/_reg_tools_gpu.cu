/*
 *  _reg_tools_gpu.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "CudaCommon.hpp"
#include "_reg_tools_gpu.h"
#include "_reg_tools_kernels.cu"

/* *************************************************************** */
void reg_voxelCentric2NodeCentric_gpu(const nifti_image *nodeImage,
                                      const nifti_image *voxelImage,
                                      float4 *nodeImageCuda,
                                      float4 *voxelImageCuda,
                                      float weight,
                                      const mat44 *voxelToMillimetre) {
    const bool is3d = nodeImage->nz > 1;
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(nodeImage, 3);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(voxelImage, 3);
    const int3 nodeImageDims = make_int3(nodeImage->nx, nodeImage->ny, nodeImage->nz);
    const int3 voxelImageDims = make_int3(voxelImage->nx, voxelImage->ny, voxelImage->nz);

    auto voxelImageTexture = Cuda::CreateTextureObject(voxelImageCuda, cudaResourceTypeLinear,
                                                       voxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);

    // The transformation between the image and the grid
    mat44 transformation;
    // Voxel to millimetre in the grid image
    if (nodeImage->sform_code > 0)
        transformation = nodeImage->sto_xyz;
    else transformation = nodeImage->qto_xyz;
    // Affine transformation between the grid and the reference image
    if (nodeImage->num_ext > 0 && nodeImage->ext_list[0].edata) {
        mat44 temp = *(reinterpret_cast<mat44*>(nodeImage->ext_list[0].edata));
        temp = nifti_mat44_inverse(temp);
        transformation = reg_mat44_mul(&temp, &transformation);
    }
    // Millimetre to voxel in the reference image
    if (voxelImage->sform_code > 0)
        transformation = reg_mat44_mul(&voxelImage->sto_ijk, &transformation);
    else transformation = reg_mat44_mul(&voxelImage->qto_ijk, &transformation);

    // The information has to be reoriented
    // Voxel to millimetre contains the orientation of the image that is used
    // to compute the spatial gradient (floating image)
    mat33 reorientation = reg_mat44_to_mat33(voxelToMillimetre);
    if (nodeImage->num_ext > 0 && nodeImage->ext_list[0].edata) {
        mat33 temp = reg_mat44_to_mat33(reinterpret_cast<mat44*>(nodeImage->ext_list[0].edata));
        temp = nifti_mat33_inverse(temp);
        reorientation = nifti_mat33_mul(temp, reorientation);
    }
    // The information has to be weighted
    float ratio[3] = { nodeImage->dx, nodeImage->dy, nodeImage->dz };
    for (int i = 0; i < (is3d ? 3 : 2); ++i) {
        if (nodeImage->sform_code > 0) {
            ratio[i] = sqrt(Square(nodeImage->sto_xyz.m[i][0]) +
                            Square(nodeImage->sto_xyz.m[i][1]) +
                            Square(nodeImage->sto_xyz.m[i][2]));
        }
        ratio[i] /= voxelImage->pixdim[i + 1];
        weight *= ratio[i];
    }

    const unsigned blocks = CudaContext::GetBlockSize()->reg_voxelCentric2NodeCentric;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)nodeNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_voxelCentric2NodeCentric_kernel<<<gridDims, blockDims>>>(nodeImageCuda, *voxelImageTexture, (unsigned)nodeNumber, nodeImageDims,
                                                                 voxelImageDims, is3d, weight, transformation, reorientation);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(const mat44 *sourceMatrixXYZ,
                                                    const nifti_image *controlPointImage,
                                                    float4 *nmiGradientCuda) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const unsigned blocks = CudaContext::GetBlockSize()->reg_convertNMIGradientFromVoxelToRealSpace;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)nodeNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    reg_convertNMIGradientFromVoxelToRealSpace_kernel<<<gridDims, blockDims>>>(nmiGradientCuda, *sourceMatrixXYZ, (unsigned)nodeNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_gaussianSmoothing_gpu(const nifti_image *image,
                               float4 *imageCuda,
                               const float& sigma,
                               const bool smoothXYZ[8]) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    bool axisToSmooth[8];
    if (smoothXYZ == nullptr) {
        for (int i = 0; i < 8; i++) axisToSmooth[i] = true;
    } else {
        for (int i = 0; i < 8; i++) axisToSmooth[i] = smoothXYZ[i];
    }

    for (int n = 1; n < 4; n++) {
        if (axisToSmooth[n] && image->dim[n] > 1) {
            float currentSigma;
            if (sigma > 0) currentSigma = sigma / image->pixdim[n];
            else currentSigma = fabs(sigma); // voxel based if negative value
            const int radius = (int)Ceil(currentSigma * 3.0f);
            if (radius > 0) {
                const int kernelSize = 1 + radius * 2;
                float *kernel;
                NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel, kernelSize * sizeof(float)));
                float kernelSum = 0;
                for (int i = -radius; i <= radius; i++) {
                    kernel[radius + i] = (float)(exp(-((float)i * (float)i) / (2.0 * currentSigma * currentSigma)) /
                                                 (currentSigma * 2.506628274631));
                    // 2.506... = sqrt(2*pi)
                    kernelSum += kernel[radius + i];
                }
                for (int i = 0; i < kernelSize; i++)
                    kernel[i] /= kernelSum;

                float *kernelCuda;
                NR_CUDA_SAFE_CALL(cudaMalloc(&kernelCuda, kernelSize * sizeof(float)));
                NR_CUDA_SAFE_CALL(cudaMemcpy(kernelCuda, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
                NR_CUDA_SAFE_CALL(cudaFreeHost(kernel));

                float4 *smoothedImage;
                NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage, voxelNumber * sizeof(float4)));

                auto imageTexture = Cuda::CreateTextureObject(imageCuda, cudaResourceTypeLinear,
                                                              voxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);
                auto kernelTexture = Cuda::CreateTextureObject(kernelCuda, cudaResourceTypeLinear,
                                                               kernelSize * sizeof(float), cudaChannelFormatKindFloat, 1);

                unsigned blocks, grids;
                dim3 blockDims, gridDims;
                switch (n) {
                case 1:
                    blocks = blockSize->reg_ApplyConvolutionWindowAlongX;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    reg_applyConvolutionWindowAlongX_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                     kernelSize, imageDim, (unsigned)voxelNumber);
                    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                    break;
                case 2:
                    blocks = blockSize->reg_ApplyConvolutionWindowAlongY;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    reg_applyConvolutionWindowAlongY_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                     kernelSize, imageDim, (unsigned)voxelNumber);
                    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                    break;
                case 3:
                    blocks = blockSize->reg_ApplyConvolutionWindowAlongZ;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    reg_applyConvolutionWindowAlongZ_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                     kernelSize, imageDim, (unsigned)voxelNumber);
                    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                    break;
                }
                NR_CUDA_SAFE_CALL(cudaFree(kernelCuda));
                NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda, smoothedImage, voxelNumber * sizeof(float4), cudaMemcpyDeviceToDevice));
                NR_CUDA_SAFE_CALL(cudaFree(smoothedImage));
            }
        }
    }
}
/* *************************************************************** */
void reg_smoothImageForCubicSpline_gpu(const nifti_image *image,
                                       float4 *imageCuda,
                                       const float *spacingVoxel) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    for (int n = 0; n < 3; n++) {
        if (spacingVoxel[n] > 0 && image->dim[n + 1] > 1) {
            int radius = Ceil(2.0 * spacingVoxel[n]);
            int kernelSize = 1 + radius * 2;

            float *kernel;
            NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel, kernelSize * sizeof(float)));

            float coeffSum = 0;
            for (int it = -radius; it <= radius; it++) {
                float coeff = (float)(fabs((float)(float)it / (float)spacingVoxel[0]));
                if (coeff < 1.0) kernel[it + radius] = (float)(2.0 / 3.0 - coeff * coeff + 0.5 * coeff * coeff * coeff);
                else if (coeff < 2.0) kernel[it + radius] = (float)(-(coeff - 2.0) * (coeff - 2.0) * (coeff - 2.0) / 6.0);
                else kernel[it + radius] = 0;
                coeffSum += kernel[it + radius];
            }
            for (int it = 0; it < kernelSize; it++)
                kernel[it] /= coeffSum;

            float *kernelCuda;
            NR_CUDA_SAFE_CALL(cudaMalloc(&kernelCuda, kernelSize * sizeof(float)));
            NR_CUDA_SAFE_CALL(cudaMemcpy(kernelCuda, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
            NR_CUDA_SAFE_CALL(cudaFreeHost(kernel));

            auto imageTexture = Cuda::CreateTextureObject(imageCuda, cudaResourceTypeLinear,
                                                          voxelNumber * sizeof(float4), cudaChannelFormatKindFloat, 4);
            auto kernelTexture = Cuda::CreateTextureObject(kernelCuda, cudaResourceTypeLinear,
                                                           kernelSize * sizeof(float), cudaChannelFormatKindFloat, 1);

            float4 *smoothedImage;
            NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage, voxelNumber * sizeof(float4)));

            unsigned grids, blocks;
            dim3 blockDims, gridDims;
            switch (n) {
            case 0:
                blocks = blockSize->reg_ApplyConvolutionWindowAlongX;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                reg_applyConvolutionWindowAlongX_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                 kernelSize, imageDim, (unsigned)voxelNumber);
                NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                break;
            case 1:
                blocks = blockSize->reg_ApplyConvolutionWindowAlongY;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                reg_applyConvolutionWindowAlongY_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                 kernelSize, imageDim, (unsigned)voxelNumber);
                NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                break;
            case 2:
                blocks = blockSize->reg_ApplyConvolutionWindowAlongZ;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                reg_applyConvolutionWindowAlongZ_kernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                 kernelSize, imageDim, (unsigned)voxelNumber);
                NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                break;
            }
            NR_CUDA_SAFE_CALL(cudaFree(kernelCuda));
            NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda, smoothedImage, voxelNumber * sizeof(float4), cudaMemcpyDeviceToDevice));
            NR_CUDA_SAFE_CALL(cudaFree(smoothedImage));
        }
    }
}
/* *************************************************************** */
void reg_multiplyValue_gpu(const size_t& count, float4 *arrayCuda, const float& value) {
    const unsigned blocks = CudaContext::GetBlockSize()->Arithmetic;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)count / (float)blocks));
    const dim3 gridDims = dim3(grids, grids, 1);
    const dim3 blockDims = dim3(blocks, 1, 1);
    reg_multiplyValue_kernel_float4<<<gridDims, blockDims>>>(arrayCuda, value, (unsigned)count);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_addValue_gpu(const size_t& count, float4 *arrayCuda, const float& value) {
    const unsigned blocks = CudaContext::GetBlockSize()->Arithmetic;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)count / (float)blocks));
    const dim3 gridDims = dim3(grids, grids, 1);
    const dim3 blockDims = dim3(blocks, 1, 1);
    reg_addValue_kernel_float4<<<gridDims, blockDims>>>(arrayCuda, value, (unsigned)count);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_multiplyArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda) {
    const unsigned blocks = CudaContext::GetBlockSize()->Arithmetic;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)count / (float)blocks));
    const dim3 gridDims = dim3(grids, grids, 1);
    const dim3 blockDims = dim3(blocks, 1, 1);
    reg_multiplyArrays_kernel_float4<<<gridDims, blockDims>>>(array1Cuda, array2Cuda, (unsigned)count);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_addArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda) {
    const unsigned blocks = CudaContext::GetBlockSize()->Arithmetic;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)count / (float)blocks));
    const dim3 gridDims = dim3(grids, grids, 1);
    const dim3 blockDims = dim3(blocks, 1, 1);
    reg_addArrays_kernel_float4<<<gridDims, blockDims>>>(array1Cuda, array2Cuda, (unsigned)count);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
float reg_sumReduction_gpu(float *arrayCuda, const size_t& size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::plus<float>());
}
/* *************************************************************** */
float reg_maxReduction_gpu(float *arrayCuda, const size_t& size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::maximum<float>());
}
/* *************************************************************** */
float reg_minReduction_gpu(float *arrayCuda, const size_t& size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::minimum<float>());
}
/* *************************************************************** */
template<typename Operation>
void reg_operationOnImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda, Operation operation) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    thrust::transform(thrust::device, img1Cuda, img1Cuda + voxelNumber, img2Cuda, img1Cuda, operation);
}
/* *************************************************************** */
void reg_addImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    reg_operationOnImages_gpu(img, img1Cuda, img2Cuda, thrust::plus<float4>());
}
/* *************************************************************** */
void reg_subtractImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    reg_operationOnImages_gpu(img, img1Cuda, img2Cuda, thrust::minus<float4>());
}
/* *************************************************************** */
void reg_multiplyImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    reg_operationOnImages_gpu(img, img1Cuda, img2Cuda, thrust::multiplies<float4>());
}
/* *************************************************************** */
void reg_divideImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    reg_operationOnImages_gpu(img, img1Cuda, img2Cuda, thrust::divides<float4>());
}
/* *************************************************************** */
DEVICE static float Min(const float& lhs, const float& rhs) {
    return lhs < rhs ? lhs : rhs;
}
DEVICE static float Max(const float& lhs, const float& rhs) {
    return lhs > rhs ? lhs : rhs;
}
using MinMaxFunc = decltype(&Min);
__device__ static MinMaxFunc minCuda = Min;
__device__ static MinMaxFunc maxCuda = Max;
/* *************************************************************** */
float reg_getMinMaxValue_gpu(const nifti_image *img, const float4 *imgCuda, const int timePoint, const bool calcMin) {
    if (timePoint < -1 || timePoint >= img->nt)
        NR_FATAL_ERROR("The required time point does not exist");

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    const int timePoints = std::clamp(timePoint > -1 ? timePoint : int(NiftiImage::calcVoxelNumber(img, 7) / voxelNumber), 1, 4);
    const float initValue = calcMin ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
    float4 result{ initValue, initValue, initValue, initValue };

    // Set the min/max functions
    MinMaxFunc minMaxCuda, minMax = calcMin ? Min : Max;
    cudaMemcpyFromSymbol(&minMaxCuda, calcMin ? minCuda : maxCuda, sizeof(MinMaxFunc));

    result = thrust::reduce(thrust::device, imgCuda, imgCuda + voxelNumber, make_float4(initValue, initValue, initValue, initValue),
                            [=]DEVICE(const float4& lhs, const float4& rhs) {
        float4 result{ initValue, initValue, initValue, initValue };
        switch (timePoints) {
        case 4:
            result.w = minMaxCuda(lhs.w, rhs.w);
            if (timePoint > -1) break;
        case 3:
            result.z = minMaxCuda(lhs.z, rhs.z);
            if (timePoint > -1) break;
        case 2:
            result.y = minMaxCuda(lhs.y, rhs.y);
            if (timePoint > -1) break;
        case 1:
            result.x = minMaxCuda(lhs.x, rhs.x);
        }
        return result;
    });

    return minMax(minMax(result.x, result.y), minMax(result.z, result.w));
}
/* *************************************************************** */
float reg_getMinValue_gpu(const nifti_image *img, const float4 *imgCuda, const int timePoint) {
    return reg_getMinMaxValue_gpu(img, imgCuda, timePoint, true);
}
/* *************************************************************** */
float reg_getMaxValue_gpu(const nifti_image *img, const float4 *imgCuda, const int timePoint) {
    return reg_getMinMaxValue_gpu(img, imgCuda, timePoint, false);
}
/* *************************************************************** */
