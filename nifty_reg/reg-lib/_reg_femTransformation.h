/*
 *  _reg_femTransformation_gpu.h
 *
 *
 *  Created by Marc Modat on 02/11/2011.
 *  Copyright (c) 2011, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_FEMTRANSFORMATION_H
#define _REG_FEMTRANSFORMATION_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>
#include "_reg_maths.h"

void reg_fem_InitialiseTransformation(int *elementNodes,
                                      unsigned int elementNumber,
                                      float *nodePositions,
                                      nifti_image *deformationFieldImage,
                                      unsigned int *closestNodes,
                                      float *femInterpolationWeight
                                      );

void reg_fem_getDeformationField(float *nodePositions,
                                 nifti_image *deformationFieldImage,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight
                                 );

void reg_fem_voxelToNodeGradient(nifti_image *voxelBasedGradient,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight,
                                 unsigned int nodeNumber,
                                 float *femBasedGradient);
#endif
