/*
 *  _reg_tools.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_H
#define _REG_TOOLS_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>

extern "C++"
void reg_intensityRescale(	nifti_image *image,
							float newMin,
							float newMax
		 				);

extern "C++" template <class PrecisionTYPE>
void reg_smoothImageForCubicSpline(	nifti_image *image,
							 int radius[3]
								  );

extern "C++" template <class PrecisionTYPE>
void reg_smoothImageForTrilinear(	nifti_image *image,
								int radius[3]
								);

extern "C++" template <class PrecisionTYPE>
void reg_gaussianSmoothing(	nifti_image *image,
						  float sigma);

extern "C++" template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image);

extern "C++" template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image);

extern "C++" template <class NewTYPE>
void reg_changeDatatype(nifti_image *image);

extern "C++"
double reg_tool_GetIntensityValue(nifti_image *,
								 int *);

extern "C++"
void reg_tools_addImages(	nifti_image *,
							nifti_image *,
							nifti_image *);

#endif
