/*
 *  _reg_affineTransformation_gpu.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANSFORMATION_H
#define _REG_AFFINETRANSFORMATION_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>
#include "_reg_maths.h"


/** reg_affine_deformationField
 * This Function compute a position field in the reference of a target image
 * using an affine transformation
 * The output position are in real space
 */
extern "C++"
void reg_affine_positionField(mat44 *,
				nifti_image *,
				nifti_image *);

extern "C++"
void reg_tool_ReadAffineFile(	mat44 *,
                                nifti_image *,
                                nifti_image *,
                                char *,
                                bool);
extern "C++"
void reg_tool_ReadAffineFile(	mat44 *,
                                char *);
extern "C++"
void reg_tool_WriteAffineFile(	mat44 *mat,
                                char *fileName);

#endif
