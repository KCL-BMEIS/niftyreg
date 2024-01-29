/*
 *  CudaGlobalTransformation.hpp
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

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool compose=false>
void GetAffineDeformationField(const mat44 *affineMatrix,
                               const nifti_image *deformationField,
                               float4 *deformationFieldCuda);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
