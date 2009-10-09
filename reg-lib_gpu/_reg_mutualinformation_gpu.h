/*
 *  _reg_mutualinformation_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_GPU_H
#define _REG_MUTUALINFORMATION_GPU_H

#include "_reg_blocksize_gpu.h"

extern "C++"
void reg_getVoxelBasedNMIGradientUsingPW_gpu(   nifti_image *targetImage,
                                                nifti_image *resultImage,
                                                float **targetImageArray_d,
                                                float **resultImageArray_d,
                                                float4 **resultGradientArray_d,
                                                float **logJointHistogram_d,
                                                float4 **voxelNMIGradientArray_d,
                                                int **targetMask_d,
                                                int activeVoxelNumber,
                                                double *entropies,
                                                int binning);


extern "C++"
void reg_smoothImageForCubicSpline_gpu(	nifti_image *resultImage,
					float4 **voxelNMIGradientArray_d,
					int *smoothingRadius);
#endif
