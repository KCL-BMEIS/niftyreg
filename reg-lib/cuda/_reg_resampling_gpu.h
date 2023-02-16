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

#include "_reg_common_cuda.h"
#include "_reg_blocksize_gpu.h"

extern "C++"
void reg_resampleImage_gpu(nifti_image *sourceImage,
                           float *resultImageArray_d,
                           cudaArray *sourceImageArray_d,
                           float4 *positionFieldImageArray_d,
                           int *mask_d,
                           size_t activeVoxelNumber,
                           float paddingValue);

extern "C++"
void reg_getImageGradient_gpu(nifti_image *sourceImage,
                              cudaArray *sourceImageArray_d,
                              float4 *positionFieldImageArray_d,
                              float4 *resultGradientArray_d,
                              size_t activeVoxelNumber,
                              float paddingValue);
