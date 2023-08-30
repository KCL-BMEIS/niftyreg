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
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

/* *************************************************************** */
extern "C++"
void reg_voxelCentric2NodeCentric_gpu(const nifti_image *nodeImage,
                                      const nifti_image *voxelImage,
                                      float4 *nodeImageCuda,
                                      float4 *voxelImageCuda,
                                      float weight,
                                      const mat44 *voxelToMillimetre);
/* *************************************************************** */
extern "C++"
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(const mat44 *sourceMatrixXYZ,
                                                    const nifti_image *controlPointImage,
                                                    float4 *nmiGradientCuda);
/* *************************************************************** */
extern "C++"
void reg_gaussianSmoothing_gpu(const nifti_image *image,
                               float4 *imageCuda,
                               const float& sigma,
                               const bool axisToSmooth[8]);
/* *************************************************************** */
extern "C++"
void reg_smoothImageForCubicSpline_gpu(const nifti_image *image,
                                       float4 *imageCuda,
                                       const float *smoothingRadius);
/* *************************************************************** */
extern "C++"
void reg_multiplyValue_gpu(const size_t& count, float4 *arrayCuda, const float& value);
/* *************************************************************** */
extern "C++"
void reg_addValue_gpu(const size_t& count, float4 *arrayCuda, const float& value);
/* *************************************************************** */
extern "C++"
void reg_multiplyArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda);
/* *************************************************************** */
extern "C++"
void reg_addArrays_gpu(const size_t& count, float4 *array1Cuda, float4 *array2Cuda);
/* *************************************************************** */
extern "C++"
void reg_fillMaskArray_gpu(int *arrayCuda, const size_t& count);
/* *************************************************************** */
extern "C++"
float reg_sumReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
extern "C++"
float reg_maxReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
extern "C++"
float reg_minReduction_gpu(float *arrayCuda, const size_t& size);
/* *************************************************************** */
