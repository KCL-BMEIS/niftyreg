/*
 * @file CudaTools.hpp
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
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool is3d>
void VoxelCentricToNodeCentric(const nifti_image *nodeImage,
                               const nifti_image *voxelImage,
                               float4 *nodeImageCuda,
                               float4 *voxelImageCuda,
                               float weight,
                               const mat44 *voxelToMillimetre = nullptr);
/* *************************************************************** */
void ConvertNmiGradientFromVoxelToRealSpace(const mat44 *sourceMatrixXYZ,
                                            const nifti_image *controlPointImage,
                                            float4 *nmiGradientCuda);
/* *************************************************************** */
void GaussianSmoothing(const nifti_image *image,
                       float4 *imageCuda,
                       const float sigma,
                       const bool axisToSmooth[8]);
/* *************************************************************** */
void SmoothImageForCubicSpline(const nifti_image *image,
                               float4 *imageCuda,
                               const float *smoothingRadius);
/* *************************************************************** */
void AddValue(const size_t count, float4 *arrayCuda, const float value);
/* *************************************************************** */
void MultiplyValue(const size_t count, float4 *arrayCuda, const float value);
/* *************************************************************** */
void MultiplyValue(const size_t count, const float4 *arrayCuda, float4 *arrayOutCuda, const float value);
/* *************************************************************** */
float SumReduction(float *arrayCuda, const size_t size);
/* *************************************************************** */
float MaxReduction(float *arrayCuda, const size_t size);
/* *************************************************************** */
float MinReduction(float *arrayCuda, const size_t size);
/* *************************************************************** */
void AddImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void SubtractImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void MultiplyImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
void DivideImages(const nifti_image *img, float4 *img1Cuda, const float4 *img2Cuda);
/* *************************************************************** */
float GetMinValue(const nifti_image *img, const float4 *imgCuda, const int timePoint = -1);
/* *************************************************************** */
float GetMaxValue(const nifti_image *img, const float4 *imgCuda, const int timePoint = -1);
/* *************************************************************** */
void SetGradientToZero(float4 *gradCuda,
                       const size_t voxelNumber,
                       const bool xAxis,
                       const bool yAxis,
                       const bool zAxis);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
