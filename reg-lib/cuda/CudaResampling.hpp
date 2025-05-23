/*
 *  CudaResampling.hpp
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "CudaCommon.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool is3d>
void ResampleImage(const nifti_image *floatingImage,
                   const float *floatingImageCuda,
                   const nifti_image *warpedImage,
                   float *warpedImageCuda,
                   const nifti_image *deformationField,
                   const float4 *deformationFieldCuda,
                   const int *maskCuda,
                   const size_t activeVoxelNumber,
                   const int interpolation,
                   const float paddingValue);
/* *************************************************************** */
template<bool is3d>
void GetImageGradient(const nifti_image *floatingImage,
                      const float *floatingImageCuda,
                      const float4 *deformationFieldCuda,
                      const nifti_image *warpedGradient,
                      float4 *warpedGradientCuda,
                      const int interpolation,
                      float paddingValue,
                      const int activeTimePoint);
/* *************************************************************** */
template<bool is3d>
void ResampleGradient(const nifti_image *floatingImage,
                      const float4 *floatingImageCuda,
                      const nifti_image *warpedImage,
                      float4 *warpedImageCuda,
                      const nifti_image *deformationField,
                      const float4 *deformationFieldCuda,
                      const int *maskCuda,
                      const size_t activeVoxelNumber,
                      const int interpolation,
                      const float paddingValue);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
