/*
 *  CudaTools.cu
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
#include "CudaTools.hpp"
#include "CudaToolsKernels.cu"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool is3d>
void VoxelCentricToNodeCentric(const nifti_image *nodeImage,
                               const nifti_image *voxelImage,
                               float4 *nodeImageCuda,
                               float4 *voxelImageCuda,
                               float weight,
                               const mat44 *voxelToMillimetre) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(nodeImage, 3);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(voxelImage, 3);
    const int3 nodeImageDims = make_int3(nodeImage->nx, nodeImage->ny, nodeImage->nz);
    const int3 voxelImageDims = make_int3(voxelImage->nx, voxelImage->ny, voxelImage->nz);
    auto voxelImageTexturePtr = Cuda::CreateTextureObject(voxelImageCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto voxelImageTexture = *voxelImageTexturePtr;

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
    transformation = reg_mat44_mul(voxelImage->sform_code > 0 ? &voxelImage->sto_ijk : &voxelImage->qto_ijk, &transformation);

    // The information has to be reoriented
    // Voxel to millimetre contains the orientation of the image that is used
    // to compute the spatial gradient (floating image)
    mat33 reorientation;
    if (voxelToMillimetre) {
        reorientation = reg_mat44_to_mat33(voxelToMillimetre);
        if (nodeImage->num_ext > 0 && nodeImage->ext_list[0].edata) {
            mat33 temp = reg_mat44_to_mat33(reinterpret_cast<mat44*>(nodeImage->ext_list[0].edata));
            temp = nifti_mat33_inverse(temp);
            reorientation = nifti_mat33_mul(temp, reorientation);
        }
    } else reg_mat33_eye(&reorientation);
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

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), nodeNumber, [=]__device__(const int index) {
        VoxelCentricToNodeCentricKernel<is3d>(nodeImageCuda, voxelImageTexture, nodeImageDims, voxelImageDims, weight, transformation, reorientation, index);
    });
}
template void VoxelCentricToNodeCentric<false>(const nifti_image*, const nifti_image*, float4*, float4*, float, const mat44*);
template void VoxelCentricToNodeCentric<true>(const nifti_image*, const nifti_image*, float4*, float4*, float, const mat44*);
/* *************************************************************** */
void ConvertNmiGradientFromVoxelToRealSpace(const mat44 *sourceMatrixXYZ,
                                            const nifti_image *controlPointImage,
                                            float4 *nmiGradientCuda) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    const unsigned blocks = CudaContext::GetBlockSize()->ConvertNmiGradientFromVoxelToRealSpace;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)nodeNumber / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);
    ConvertNmiGradientFromVoxelToRealSpaceKernel<<<gridDims, blockDims>>>(nmiGradientCuda, *sourceMatrixXYZ, (unsigned)nodeNumber);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void GaussianSmoothing(const nifti_image *image,
                       float4 *imageCuda,
                       const float sigma,
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

                auto imageTexture = Cuda::CreateTextureObject(imageCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
                auto kernelTexture = Cuda::CreateTextureObject(kernelCuda, kernelSize, cudaChannelFormatKindFloat, 1);

                unsigned blocks, grids;
                dim3 blockDims, gridDims;
                switch (n) {
                case 1:
                    blocks = blockSize->ApplyConvolutionWindowAlongX;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    ApplyConvolutionWindowAlongXKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                     kernelSize, imageDim, (unsigned)voxelNumber);
                    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                    break;
                case 2:
                    blocks = blockSize->ApplyConvolutionWindowAlongY;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    ApplyConvolutionWindowAlongYKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                     kernelSize, imageDim, (unsigned)voxelNumber);
                    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                    break;
                case 3:
                    blocks = blockSize->ApplyConvolutionWindowAlongZ;
                    grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                    gridDims = dim3(grids, grids, 1);
                    blockDims = dim3(blocks, 1, 1);
                    ApplyConvolutionWindowAlongZKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
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
void SmoothImageForCubicSpline(const nifti_image *image,
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

            auto imageTexture = Cuda::CreateTextureObject(imageCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
            auto kernelTexture = Cuda::CreateTextureObject(kernelCuda, kernelSize, cudaChannelFormatKindFloat, 1);

            float4 *smoothedImage;
            NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage, voxelNumber * sizeof(float4)));

            unsigned grids, blocks;
            dim3 blockDims, gridDims;
            switch (n) {
            case 0:
                blocks = blockSize->ApplyConvolutionWindowAlongX;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                ApplyConvolutionWindowAlongXKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                 kernelSize, imageDim, (unsigned)voxelNumber);
                NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                break;
            case 1:
                blocks = blockSize->ApplyConvolutionWindowAlongY;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                ApplyConvolutionWindowAlongYKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
                                                                                 kernelSize, imageDim, (unsigned)voxelNumber);
                NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
                break;
            case 2:
                blocks = blockSize->ApplyConvolutionWindowAlongZ;
                grids = (unsigned)Ceil(sqrtf((float)voxelNumber / (float)blocks));
                gridDims = dim3(grids, grids, 1);
                blockDims = dim3(blocks, 1, 1);
                ApplyConvolutionWindowAlongZKernel<<<gridDims, blockDims>>>(smoothedImage, *imageTexture, *kernelTexture,
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
void AddValue(const size_t count, float4 *arrayCuda, const float addition) {
    thrust::for_each_n(thrust::device, arrayCuda, count, [=]__device__(float4& val) {
        val = val + addition;
    });
}
/* *************************************************************** */
void MultiplyValue(const size_t count, float4 *arrayCuda, const float multiplier) {
    thrust::for_each_n(thrust::device, arrayCuda, count, [=]__device__(float4& val) {
        val = val * multiplier;
    });
}
/* *************************************************************** */
void MultiplyValue(const size_t count, const float4 *arrayCuda, float4 *arrayOutCuda, const float multiplier) {
    auto arrayTexturePtr = Cuda::CreateTextureObject(arrayCuda, count, cudaChannelFormatKindFloat, 4);
    auto arrayTexture = *arrayTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), count, [=]__device__(const int index) {
        float4 val = tex1Dfetch<float4>(arrayTexture, index);
        arrayOutCuda[index] = val * multiplier;
    });
}
/* *************************************************************** */
float SumReduction(float *arrayCuda, const size_t size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::plus<float>());
}
/* *************************************************************** */
float MaxReduction(float *arrayCuda, const size_t size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::maximum<float>());
}
/* *************************************************************** */
float MinReduction(float *arrayCuda, const size_t size) {
    thrust::device_ptr<float> dptr(arrayCuda);
    return thrust::reduce(thrust::device, dptr, dptr + size, 0.f, thrust::minimum<float>());
}
/* *************************************************************** */
template<typename Operation>
void OperationOnImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda, Operation operation) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    thrust::transform(thrust::device, img1Cuda, img1Cuda + voxelNumber, img2Cuda, img1Cuda, operation);
}
/* *************************************************************** */
void AddImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    OperationOnImages(img, img1Cuda, img2Cuda, thrust::plus<float4>());
}
/* *************************************************************** */
void SubtractImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    OperationOnImages(img, img1Cuda, img2Cuda, thrust::minus<float4>());
}
/* *************************************************************** */
void MultiplyImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    OperationOnImages(img, img1Cuda, img2Cuda, thrust::multiplies<float4>());
}
/* *************************************************************** */
void DivideImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda) {
    OperationOnImages(img, img1Cuda, img2Cuda, thrust::divides<float4>());
}
/* *************************************************************** */
template<bool isMin>
DEVICE static inline float MinMax(const float lhs, const float rhs) {
    if constexpr (isMin) return lhs < rhs ? lhs : rhs;
    else return lhs > rhs ? lhs : rhs;
}
/* *************************************************************** */
template<bool isMin, bool isSingleTimePoint, int timePoints>
inline float GetMinMaxValue(const nifti_image *img, const float4 *imgCuda) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    constexpr float initVal = isMin ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();

    const float4 result = thrust::reduce(thrust::device, imgCuda, imgCuda + voxelNumber, make_float4(initVal, initVal, initVal, initVal),
                                         [=]DEVICE(const float4& lhs, const float4& rhs) {
        float4 result{ initVal, initVal, initVal, initVal };
        switch (timePoints) {
        case 4:
            result.w = MinMax<isMin>(lhs.w, rhs.w);
            if constexpr (isSingleTimePoint) break;
        case 3:
            result.z = MinMax<isMin>(lhs.z, rhs.z);
            if constexpr (isSingleTimePoint) break;
        case 2:
            result.y = MinMax<isMin>(lhs.y, rhs.y);
            if constexpr (isSingleTimePoint) break;
        case 1:
            result.x = MinMax<isMin>(lhs.x, rhs.x);
        }
        return result;
    });

    return MinMax<isMin>(MinMax<isMin>(result.x, result.y), MinMax<isMin>(result.z, result.w));
}
/* *************************************************************** */
template<bool isMin, bool isSingleTimePoint>
static inline float GetMinMaxValue(const nifti_image *img, const float4 *imgCuda, const int timePoints) {
    auto getMinMaxValue = GetMinMaxValue<isMin, isSingleTimePoint, 1>;
    switch (timePoints) {
    case 2:
        getMinMaxValue = GetMinMaxValue<isMin, isSingleTimePoint, 2>;
        break;
    case 3:
        getMinMaxValue = GetMinMaxValue<isMin, isSingleTimePoint, 3>;
        break;
    case 4:
        getMinMaxValue = GetMinMaxValue<isMin, isSingleTimePoint, 4>;
        break;
    }
    return getMinMaxValue(img, imgCuda);
}
/* *************************************************************** */
template<bool isMin>
static inline float GetMinMaxValue(const nifti_image *img, const float4 *imgCuda, const int timePoint) {
    if (timePoint < -1 || timePoint >= img->nt)
        NR_FATAL_ERROR("The required time point does not exist");
    const bool isSingleTimePoint = timePoint > -1;
    const int timePoints = std::clamp(isSingleTimePoint ? timePoint + 1 : img->nt * img->nu, 1, 4);
    auto getMinMaxValue = GetMinMaxValue<isMin, false>;
    if (isSingleTimePoint) getMinMaxValue = GetMinMaxValue<isMin, true>;
    return getMinMaxValue(img, imgCuda, timePoints);
}
/* *************************************************************** */
float GetMinValue(const nifti_image *img, const float4 *imgCuda, const int timePoint) {
    return GetMinMaxValue<true>(img, imgCuda, timePoint);
}
/* *************************************************************** */
float GetMaxValue(const nifti_image *img, const float4 *imgCuda, const int timePoint) {
    return GetMinMaxValue<false>(img, imgCuda, timePoint);
}
/* *************************************************************** */
template<bool xAxis, bool yAxis, bool zAxis>
void SetGradientToZero(float4 *gradCuda, const size_t voxelNumber) {
    auto gradTexturePtr = Cuda::CreateTextureObject(gradCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto gradTexture = *gradTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), voxelNumber, [gradCuda, gradTexture]__device__(const int index) {
        if constexpr (xAxis && yAxis && zAxis) {
            gradCuda[index] = make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
            float4 val = tex1Dfetch<float4>(gradTexture, index);
            if constexpr (xAxis) val.x = 0;
            if constexpr (yAxis) val.y = 0;
            if constexpr (zAxis) val.z = 0;
            gradCuda[index] = val;
        }
    });
}
/* *************************************************************** */
void SetGradientToZero(float4 *gradCuda, const size_t voxelNumber, const bool xAxis, const bool yAxis, const bool zAxis) {
    if (!xAxis && !yAxis && !zAxis) return;
    decltype(SetGradientToZero<true, true, true>) *setGradientToZero;
    if (xAxis && yAxis && zAxis) setGradientToZero = SetGradientToZero<true, true, true>;
    else if (xAxis && yAxis) setGradientToZero = SetGradientToZero<true, true, false>;
    else if (xAxis && zAxis) setGradientToZero = SetGradientToZero<true, false, true>;
    else if (yAxis && zAxis) setGradientToZero = SetGradientToZero<false, true, true>;
    else if (xAxis) setGradientToZero = SetGradientToZero<true, false, false>;
    else if (yAxis) setGradientToZero = SetGradientToZero<false, true, false>;
    else if (zAxis) setGradientToZero = SetGradientToZero<false, false, true>;
    setGradientToZero(gradCuda, voxelNumber);
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
