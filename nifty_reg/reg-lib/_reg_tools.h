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
void reg_checkAndCorrectDimension(nifti_image *image);

extern "C++"
void reg_intensityRescale(	nifti_image *image,
                            float *newMin,
                            float *newMax,
                            float *lowThr,
                            float *upThr
		 				);

extern "C++" template <class PrecisionTYPE>
void reg_smoothImageForCubicSpline(nifti_image *image,
                                   int radius[3]
                                   );

extern "C++" template <class PrecisionTYPE>
void reg_smoothImageForTrilinear(	nifti_image *image,
								int radius[3]
								);

extern "C++" template <class PrecisionTYPE>
void reg_gaussianSmoothing(	nifti_image *image,
                            PrecisionTYPE sigma,
                            bool[8]);

extern "C++" template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image, int, bool[8]);

extern "C++" template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image);

extern "C++" template <class NewTYPE>
void reg_changeDatatype(nifti_image *image);

extern "C++"
double reg_tool_GetIntensityValue(nifti_image *,
                                  int *);

extern "C++"
void reg_tools_addSubMulDivImages(  nifti_image *,
                                    nifti_image *,
                                    nifti_image *,
                                    int);
extern "C++"
void reg_tools_addSubMulDivValue(  nifti_image *,
                                    nifti_image *,
                                    float,
                                    int);

extern "C++"
void reg_tool_binarise_image(nifti_image *);

extern "C++"
void reg_tool_binarise_image(nifti_image *, float);

extern "C++"
void reg_tool_binaryImage2int(nifti_image *, int *, int &);

extern "C++"
double reg_tools_getMeanRMS(nifti_image *, nifti_image *);

extern "C++"
int reg_tool_nanMask_image(nifti_image *, nifti_image *, nifti_image *);

/** JM functions for ssd */
//this function will threshold an image to the values provided,
//set the scl_slope and sct_inter of the image to 1 and 0 (SSD uses actual image data values),
//and sets cal_min and cal_max to have the min/max image data values
extern "C++" template<class T>
void reg_thresholdImage(nifti_image *image,
                            T lowThr,
                            T upThr
		 				);
#endif
