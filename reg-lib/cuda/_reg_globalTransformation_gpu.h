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

#ifndef _REG_AFFINETRANSFORMATION_GPU_H
#define _REG_AFFINETRANSFORMATION_GPU_H

#include "_reg_common_gpu.h"
#include "_reg_globalTransformation.h"

extern "C++"
void reg_affine_positionField_gpu(mat44 *,
                                  nifti_image *,
                                  float4 **);

#endif
