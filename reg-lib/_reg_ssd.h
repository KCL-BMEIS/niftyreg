/*
 *  _reg_ssd.h
 *  
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_H
#define _REG_SSD_H

#include "nifti1_io.h"

extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_getSSD(	nifti_image *targetImage,
	 						nifti_image *resultImage
 							);

extern "C++" template <class PrecisionTYPE>
void reg_getVoxelBasedSSDGradient(	PrecisionTYPE SSDValue,
									nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									nifti_image *ssdGradientImage);
#endif
