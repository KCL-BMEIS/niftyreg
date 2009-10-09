/*
 *  _reg_mutualinformation.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_H
#define _REG_MUTUALINFORMATION_H

#include "nifti1_io.h"

extern "C++" template<class PrecisionTYPE>
void reg_getEntropies(	nifti_image *targetImage,
						nifti_image *resultImage,
						int type,
						int binning,
						PrecisionTYPE *probaJointHistogram,
						PrecisionTYPE *logJointHistogram,
						PrecisionTYPE *entropies,
                        int *mask);

extern "C++" template <class PrecisionTYPE>
void reg_getVoxelBasedNMIGradientUsingPW(	nifti_image *targetImage,
											nifti_image *resultImage,
                                            int type,
											nifti_image *resultImageGradient,
											int binning,
											PrecisionTYPE *logJointHistogram,
											PrecisionTYPE *entropies,
											nifti_image *nmiGradientImage,
                                            int *mask);
#endif
