/*
 *  _reg_resampling_gpu.h
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
void reg_resampleImage_gpu(const nifti_image *floatingImage,
                           float *warpedImageCuda,
                           const cudaArray *floatingImageCuda,
                           const float4 *deformationFieldCuda,
                           const int *maskCuda,
                           const size_t& activeVoxelNumber,
                           const float& paddingValue);
/* *************************************************************** */
void reg_getImageGradient_gpu(const nifti_image *floatingImage,
                              const cudaArray *floatingImageCuda,
                              const float4 *deformationFieldCuda,
                              float4 *warpedGradientCuda,
                              const size_t& activeVoxelNumber,
                              const float& paddingValue);
/* *************************************************************** */
