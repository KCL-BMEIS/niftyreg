/*
 * @file _reg_tools_gpu.h
 * @author Marc Modat
 * @date 24/03/2009
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "CudaCommon.hpp"
#include "_reg_tools.h"

/* *************************************************************** */
void reg_voxelCentric2NodeCentric_gpu(const nifti_image *nodeImage,
                                      const nifti_image *voxelImage,
                                      float4 *nodeImageCuda,
                                      float4 *voxelImageCuda,
                                      float weight,
                                      const mat44 *voxelToMillimetre);
/* *************************************************************** */
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(const mat44 *sourceMatrixXYZ,
                                                    const nifti_image *controlPointImage,
                                                    float4 *nmiGradientCuda);
/* *************************************************************** */
void reg_gaussianSmoothing_gpu(const nifti_image *image,
                               float4 *imageCuda,
                               const float& sigma,
                               const bool axisToSmooth[8]);
/* *************************************************************** */
void reg_smoothImageForCubicSpline_gpu(const nifti_image *image,
                                       float4 *imageCuda,
                                       const float *smoothingRadius);
/* *************************************************************** */
void reg_multiplyValue_gpu(const size_t& count, float4 *arrayCuda, const float& value);
/* *************************************************************** */
void reg_addValue_gpu(const size_t& count, float4 *arrayCuda, const float& value);
/* *************************************************************** */
void reg_multiplyArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda);
/* *************************************************************** */
void reg_addArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda);
/* *************************************************************** */
void reg_fillMaskArray_gpu(int *arrayCuda, const size_t& count);
/* *************************************************************** */
float reg_sumReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
float reg_maxReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
float reg_minReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
void reg_addImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void reg_subtractImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void reg_multiplyImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void reg_divideImages_gpu(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
float reg_getMinValue_gpu(const nifti_image *img, const float4 *imgCuda, const int timePoint = -1);
/* *************************************************************** */
float reg_getMaxValue_gpu(const nifti_image *img, const float4 *imgCuda, const int timePoint = -1);
/* *************************************************************** */
