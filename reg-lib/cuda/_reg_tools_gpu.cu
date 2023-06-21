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

#include "_reg_common_cuda.h"
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

    auto voxelImageTexture = cudaCommon_createTextureObject(voxelImageCuda, cudaResourceTypeLinear,
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
            ratio[i] = sqrt(reg_pow2(nodeImage->sto_xyz.m[i][0]) +
                            reg_pow2(nodeImage->sto_xyz.m[i][1]) +
                            reg_pow2(nodeImage->sto_xyz.m[i][2]));
        }
        ratio[i] /= voxelImage->pixdim[i + 1];
        weight *= ratio[i];
    }

    const unsigned blocks = NiftyReg::CudaContext::GetBlockSize()->reg_voxelCentric2NodeCentric;
    const unsigned grids = (unsigned)ceil(sqrtf((float)nodeNumber / (float)blocks));
    const dim3 blockDims(blocks, 1, 1);
    const dim3 gridDims(grids, grids, 1);
    reg_voxelCentric2NodeCentric_kernel<<<gridDims, blockDims>>>(nodeImageCuda, *voxelImageTexture, (unsigned)nodeNumber, nodeImageDims,
                                                                 voxelImageDims, is3d, weight, transformation, reorientation);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(mat44 *sourceMatrix_xyz,
                                                    nifti_image *controlPointImage,
                                                    float4 *nodeNMIGradientArray_d) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    const int nodeNumber = CalcVoxelNumber(*controlPointImage);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber, &nodeNumber, sizeof(int)));

    float4 *matrix_h; NR_CUDA_SAFE_CALL(cudaMallocHost(&matrix_h, 3 * sizeof(float4)));
    matrix_h[0] = make_float4(sourceMatrix_xyz->m[0][0], sourceMatrix_xyz->m[0][1], sourceMatrix_xyz->m[0][2], sourceMatrix_xyz->m[0][3]);
    matrix_h[1] = make_float4(sourceMatrix_xyz->m[1][0], sourceMatrix_xyz->m[1][1], sourceMatrix_xyz->m[1][2], sourceMatrix_xyz->m[1][3]);
    matrix_h[2] = make_float4(sourceMatrix_xyz->m[2][0], sourceMatrix_xyz->m[2][1], sourceMatrix_xyz->m[2][2], sourceMatrix_xyz->m[2][3]);
    float4 *matrix_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&matrix_d, 3 * sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(matrix_d, matrix_h, 3 * sizeof(float4), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaFreeHost(matrix_h));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, matrixTexture, matrix_d, 3 * sizeof(float4)));

    const unsigned Grid_reg_convertNMIGradientFromVoxelToRealSpace =
        (unsigned)ceil(sqrtf((float)nodeNumber / (float)blockSize->reg_convertNMIGradientFromVoxelToRealSpace));
    dim3 G1(Grid_reg_convertNMIGradientFromVoxelToRealSpace, Grid_reg_convertNMIGradientFromVoxelToRealSpace, 1);
    dim3 B1(blockSize->reg_convertNMIGradientFromVoxelToRealSpace, 1, 1);
    _reg_convertNMIGradientFromVoxelToRealSpace_kernel<<<G1, B1>>>(nodeNMIGradientArray_d);
    NR_CUDA_CHECK_KERNEL(G1, B1);

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(matrixTexture));
    NR_CUDA_SAFE_CALL(cudaFree(matrix_d));
}
/* *************************************************************** */
void reg_gaussianSmoothing_gpu(nifti_image *image,
                               float4 *imageArray_d,
                               float sigma,
                               bool smoothXYZ[8]) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    const int voxelNumber = CalcVoxelNumber(*image);
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageDim, &imageDim, sizeof(int3)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &voxelNumber, sizeof(int)));

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
            int radius = (int)ceil(currentSigma * 3.0f);
            if (radius > 0) {
                int kernelSize = 1 + radius * 2;
                float *kernel_h;
                NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel_h, kernelSize * sizeof(float)));
                float kernelSum = 0;
                for (int i = -radius; i <= radius; i++) {
                    kernel_h[radius + i] = (float)(exp(-((float)i * (float)i) / (2.0 * currentSigma * currentSigma)) /
                                                   (currentSigma * 2.506628274631));
                    // 2.506... = sqrt(2*pi)
                    kernelSum += kernel_h[radius + i];
                }
                for (int i = 0; i < kernelSize; i++)
                    kernel_h[i] /= kernelSum;

                float *kernel_d;
                NR_CUDA_SAFE_CALL(cudaMalloc(&kernel_d, kernelSize * sizeof(float)));
                NR_CUDA_SAFE_CALL(cudaMemcpy(kernel_d, kernel_h, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
                NR_CUDA_SAFE_CALL(cudaFreeHost(kernel_h));

                float4 *smoothedImage;
                NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage, voxelNumber * sizeof(float4)));
                NR_CUDA_SAFE_CALL(cudaBindTexture(0, convolutionKernelTexture, kernel_d, kernelSize * sizeof(float)));
                NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, imageArray_d, voxelNumber * sizeof(float4)));

                unsigned Grid_reg_ApplyConvolutionWindow;
                dim3 B, G;
                switch (n) {
                case 1:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongX));
                    B = dim3(blockSize->reg_ApplyConvolutionWindowAlongX, 1, 1);
                    G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                    _reg_ApplyConvolutionWindowAlongX_kernel<<<G, B>>>(smoothedImage, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G, B);
                    break;
                case 2:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongY));
                    B = dim3(blockSize->reg_ApplyConvolutionWindowAlongY, 1, 1);
                    G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                    _reg_ApplyConvolutionWindowAlongY_kernel<<<G, B>>>(smoothedImage, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G, B);
                    break;
                case 3:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongZ));
                    B = dim3(blockSize->reg_ApplyConvolutionWindowAlongZ, 1, 1);
                    G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                    _reg_ApplyConvolutionWindowAlongZ_kernel<<<G, B>>>(smoothedImage, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G, B);
                    break;
                }
                NR_CUDA_SAFE_CALL(cudaUnbindTexture(convolutionKernelTexture));
                NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture));
                NR_CUDA_SAFE_CALL(cudaFree(kernel_d));
                NR_CUDA_SAFE_CALL(cudaMemcpy(imageArray_d, smoothedImage, voxelNumber * sizeof(float4), cudaMemcpyDeviceToDevice));
                NR_CUDA_SAFE_CALL(cudaFree(smoothedImage));
            }
        }
    }
}
/* *************************************************************** */
void reg_smoothImageForCubicSpline_gpu(nifti_image *image,
                                       float4 *imageArray_d,
                                       float *spacingVoxel) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    const int voxelNumber = CalcVoxelNumber(*image);
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageDim, &imageDim, sizeof(int3)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &voxelNumber, sizeof(int)));

    for (int n = 0; n < 3; n++) {
        if (spacingVoxel[n] > 0 && image->dim[n + 1] > 1) {
            int radius = static_cast<int>(reg_ceil(2.0 * spacingVoxel[n]));
            int kernelSize = 1 + radius * 2;

            float *kernel_h;
            NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel_h, kernelSize * sizeof(float)));

            float coeffSum = 0;
            for (int it = -radius; it <= radius; it++) {
                float coeff = (float)(fabs((float)(float)it / (float)spacingVoxel[0]));
                if (coeff < 1.0) kernel_h[it + radius] = (float)(2.0 / 3.0 - coeff * coeff + 0.5 * coeff * coeff * coeff);
                else if (coeff < 2.0) kernel_h[it + radius] = (float)(-(coeff - 2.0) * (coeff - 2.0) * (coeff - 2.0) / 6.0);
                else kernel_h[it + radius] = 0;
                coeffSum += kernel_h[it + radius];
            }
            for (int it = 0; it < kernelSize; it++) kernel_h[it] /= coeffSum;

            float *kernel_d;
            NR_CUDA_SAFE_CALL(cudaMalloc(&kernel_d, kernelSize * sizeof(float)));
            NR_CUDA_SAFE_CALL(cudaMemcpy(kernel_d, kernel_h, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
            NR_CUDA_SAFE_CALL(cudaFreeHost(kernel_h));
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, convolutionKernelTexture, kernel_d, kernelSize * sizeof(float)));

            float4 *smoothedImage_d;
            NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage_d, voxelNumber * sizeof(float4)));
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, imageArray_d, voxelNumber * sizeof(float4)));

            unsigned Grid_reg_ApplyConvolutionWindow;
            dim3 B, G;
            switch (n) {
            case 0:
                Grid_reg_ApplyConvolutionWindow =
                    (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongX));
                B = dim3(blockSize->reg_ApplyConvolutionWindowAlongX, 1, 1);
                G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                _reg_ApplyConvolutionWindowAlongX_kernel<<<G, B>>>(smoothedImage_d, kernelSize);
                NR_CUDA_CHECK_KERNEL(G, B);
                break;
            case 1:
                Grid_reg_ApplyConvolutionWindow =
                    (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongY));
                B = dim3(blockSize->reg_ApplyConvolutionWindowAlongY, 1, 1);
                G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                _reg_ApplyConvolutionWindowAlongY_kernel<<<G, B>>>(smoothedImage_d, kernelSize);
                NR_CUDA_CHECK_KERNEL(G, B);
                break;
            case 2:
                Grid_reg_ApplyConvolutionWindow =
                    (unsigned)ceil(sqrtf((float)voxelNumber / (float)blockSize->reg_ApplyConvolutionWindowAlongZ));
                B = dim3(blockSize->reg_ApplyConvolutionWindowAlongZ, 1, 1);
                G = dim3(Grid_reg_ApplyConvolutionWindow, Grid_reg_ApplyConvolutionWindow, 1);
                _reg_ApplyConvolutionWindowAlongZ_kernel<<<G, B>>>(smoothedImage_d, kernelSize);
                NR_CUDA_CHECK_KERNEL(G, B);
                break;
            }

            NR_CUDA_SAFE_CALL(cudaUnbindTexture(convolutionKernelTexture));
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture));
            NR_CUDA_SAFE_CALL(cudaFree(kernel_d));
            NR_CUDA_SAFE_CALL(cudaMemcpy(imageArray_d, smoothedImage_d, voxelNumber * sizeof(float4), cudaMemcpyDeviceToDevice));
            NR_CUDA_SAFE_CALL(cudaFree(smoothedImage_d));
        }
    }
}
/* *************************************************************** */
void reg_multiplyValue_gpu(int num, float4 *array_d, float value) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &num, sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight, &value, sizeof(float)));

    const unsigned Grid_reg_multiplyValues = (unsigned)ceil(sqrtf((float)num / (float)blockSize->reg_arithmetic));
    dim3 G = dim3(Grid_reg_multiplyValues, Grid_reg_multiplyValues, 1);
    dim3 B = dim3(blockSize->reg_arithmetic, 1, 1);
    reg_multiplyValue_kernel_float4<<<G, B>>>(array_d);
    NR_CUDA_CHECK_KERNEL(G, B);
}
/* *************************************************************** */
void reg_addValue_gpu(int num, float4 *array_d, float value) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &num, sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight, &value, sizeof(float)));

    const unsigned Grid_reg_addValues = (unsigned)ceil(sqrtf((float)num / (float)blockSize->reg_arithmetic));
    dim3 G = dim3(Grid_reg_addValues, Grid_reg_addValues, 1);
    dim3 B = dim3(blockSize->reg_arithmetic, 1, 1);
    reg_addValue_kernel_float4<<<G, B>>>(array_d);
    NR_CUDA_CHECK_KERNEL(G, B);
}
/* *************************************************************** */
void reg_multiplyArrays_gpu(int num, float4 *array1_d, float4 *array2_d) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &num, sizeof(int)));

    const unsigned Grid_reg_multiplyArrays = (unsigned)ceil(sqrtf((float)num / (float)blockSize->reg_arithmetic));
    dim3 G = dim3(Grid_reg_multiplyArrays, Grid_reg_multiplyArrays, 1);
    dim3 B = dim3(blockSize->reg_arithmetic, 1, 1);
    reg_multiplyArrays_kernel_float4<<<G, B>>>(array1_d, array2_d);
    NR_CUDA_CHECK_KERNEL(G, B);
}
/* *************************************************************** */
void reg_addArrays_gpu(int num, float4 *array1_d, float4 *array2_d) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &num, sizeof(int)));

    const unsigned Grid_reg_addArrays = (unsigned)ceil(sqrtf((float)num / (float)blockSize->reg_arithmetic));
    dim3 G = dim3(Grid_reg_addArrays, Grid_reg_addArrays, 1);
    dim3 B = dim3(blockSize->reg_arithmetic, 1, 1);
    reg_addArrays_kernel_float4<<<G, B>>>(array1_d, array2_d);
    NR_CUDA_CHECK_KERNEL(G, B);
}
/* *************************************************************** */
void reg_fillMaskArray_gpu(int num, int *array1_d) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &num, sizeof(int)));

    const unsigned Grid_reg_fillMaskArray = (unsigned)ceil(sqrtf((float)num / (float)blockSize->reg_arithmetic));
    dim3 G = dim3(Grid_reg_fillMaskArray, Grid_reg_fillMaskArray, 1);
    dim3 B = dim3(blockSize->reg_arithmetic, 1, 1);
    reg_fillMaskArray_kernel<<<G, B>>>(array1_d);
    NR_CUDA_CHECK_KERNEL(G, B);
}
/* *************************************************************** */
float reg_sumReduction_gpu(float *array_d, size_t size) {
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr, dptr + size, 0.f, thrust::plus<float>());
}
/* *************************************************************** */
float reg_maxReduction_gpu(float *array_d, size_t size) {
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr, dptr + size, 0.f, thrust::maximum<float>());
}
/* *************************************************************** */
float reg_minReduction_gpu(float *array_d, size_t size) {
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr, dptr + size, 0.f, thrust::minimum<float>());
}
/* *************************************************************** */
