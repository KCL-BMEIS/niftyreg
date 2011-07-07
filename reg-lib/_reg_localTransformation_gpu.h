/*
 *  _reg_bspline_gpu.h
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_GPU_H
#define _REG_BSPLINE_GPU_H

#include "_reg_blocksize_gpu.h"
#include "_reg_maths.h"
#include <limits>

extern "C++"
void reg_bspline_gpu(   nifti_image *controlPointImage,
                        nifti_image *targetImage,
                        float4 **controlPointImageArray_d,
                        float4 **positionFieldImageArray_d,
                        int **mask,
                        int activeVoxelNumber,
                        bool bspline);

/* BE */
extern "C++"
float reg_bspline_ApproxBendingEnergy_gpu(nifti_image *controlPointImage,
                                          float4 **controlPointImageArray_d);

extern "C++"
void reg_bspline_ApproxBendingEnergyGradient_gpu(nifti_image *referenceImage,
                                                 nifti_image *controlPointImage,
                                                 float4 **controlPointImageArray_d,
                                                 float4 **nodeNMIGradientArray_d,
                                                 float bendingEnergyWeight);

/** Jacobian
 *
 */
extern "C++"
double reg_bspline_ComputeJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
                                                  nifti_image *controlPointImage,
                                                  float4 **controlPointImageArray_d,
                                                  bool approx);

extern "C++"
void reg_bspline_ComputeJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
                                                        nifti_image *controlPointImage,
                                                        float4 **controlPointImageArray_d,
                                                        float4 **nodeNMIGradientArray_d,
                                                        float jacobianWeight,
                                                        bool approx);

extern "C++"
double reg_bspline_correctFolding_gpu(  nifti_image *targetImage,
                                        nifti_image *controlPointImage,
                                        float4 **controlPointImageArray_d,
                                        bool approx);

///** reg_spline_getDeformationFromDisplacement_gpu
// * Convert a displacement image into a deformation image using the image orientation header
// */
//extern "C++"
//void reg_spline_getDeformationFromDisplacement_gpu( nifti_image *image,
//                                                    float4 **imageArray_d);

#endif
