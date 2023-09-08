/*
 *  blockMatchingKernel.h
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "CudaCommon.hpp"
#include "_reg_blockMatching.h"

/**
 * @brief Block matching method
 * @param referenceImage The reference image.
 * @param params The block matching parameters.
 * @param referenceImageCuda The reference image on the device.
 * @param warpedImageCuda The warped image on the device.
 * @param referencePositionCuda Output. The centre of the blocks in the reference image.
 * @param warpedPositionCuda Output. The corresponding centre of the blocks in the result.
 * @param totalBlockCuda Array specifying which blocks are active.
 * @param maskCuda The mask image on the device.
 * @param refMatCuda The reference image transformation matrix on the device.
 */
void block_matching_method_gpu(const nifti_image *referenceImage,
                               _reg_blockMatchingParam *params,
                               const float *referenceImageCuda,
                               const float *warpedImageCuda,
                               float *referencePositionCuda,
                               float *warpedPositionCuda,
                               const int *totalBlockCuda,
                               const int *maskCuda,
                               const float *refMatCuda);
