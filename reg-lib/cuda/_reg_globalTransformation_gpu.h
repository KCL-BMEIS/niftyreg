/*
 *  _reg_affineTransformation.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "CudaCommon.hpp"

extern "C++"
void reg_affine_positionField_gpu(const mat44 *affineMatrix,
                                  const nifti_image *targetImage,
                                  float4 *deformationFieldCuda);
