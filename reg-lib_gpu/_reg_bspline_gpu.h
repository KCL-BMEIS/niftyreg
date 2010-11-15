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
#include <limits>

#define SCALING_VALUE 256
#define SQUARING_VALUE 8

extern "C++"
void reg_bspline_gpu(   nifti_image *controlPointImage,
                        nifti_image *targetImage,
                        float4 **controlPointImageArray_d,
                        float4 **positionFieldImageArray_d,
                        int **mask,
                        int activeVoxelNumber);

/* BE */
extern "C++"
float reg_bspline_ApproxBendingEnergy_gpu(	nifti_image *controlPointImage,
                                            float4 **controlPointImageArray_d);

extern "C++"
void reg_bspline_ApproxBendingEnergyGradient_gpu(   nifti_image *targetImage,
                                                    nifti_image *controlPointImage,
							                        float4 **controlPointImageArray_d,
							                        float4 **nodeNMIGradientArray_d,
							                        float bendingEnergyWeight);

/* Jacobian */
extern "C++"
void reg_bspline_ComputeJacobianMap(nifti_image *targetImage,
                                    nifti_image *controlPointImage,
                                    float4 **controlPointImageArray_d,
                                    float **jacobianMap);

extern "C++"
double reg_bspline_ComputeJacobianPenaltyTerm_gpu(  nifti_image *targetImage,
                                                    nifti_image *controlPointImage,
                                                    float4 **controlPointImageArray_d,
                                                    bool approximate);

extern "C++"
double reg_bspline_ComputeJacobianPenaltyTermFromVelocity_gpu(  nifti_image *targetImage,
                                                                nifti_image *velocityFieldImage,
                                                                float4 **velocityFieldImageArray_d,
                                                                bool approximate);

extern "C++"
void reg_bspline_ComputeJacobianGradient_gpu(   nifti_image *targetImage,
                                                nifti_image *controlPointImage,
                                                float4 **controlPointImageArray_d,
                                                float4 **nodeNMIGradientArray_d,
                                                float jacobianWeight,
                                                bool appJacobianFlag);

extern "C++"
void reg_bspline_ComputeJacGradientFromVelocity_gpu(nifti_image *targetImage,
                                                    nifti_image *velocityFieldImage,
                                                    float4 **velocityFieldImageArray_d,
                                                    float4 **gradientImageArray_d,
                                                    float jacobianWeight,
                                                    bool approximate);

extern "C++"
double reg_bspline_correctFolding_gpu(  nifti_image *targetImage,
                                        nifti_image *controlPointImage,
                                        float4 **controlPointImageArray_d,
                                        bool approx);

/** Composition of control point grid */
extern "C++"
void reg_spline_cppComposition_gpu( nifti_image *toUpdate,
                                    nifti_image *toCompose, // displacement or deformation
                                    float4 **toUpdateArray_d,
                                    float4 **toComposeArray_d,
                                    float ratio,
                                    bool type);

/** a control point grid is decomposed in order to get the interpolation coefficients */
extern "C++"
void reg_spline_cppDeconvolve_gpu(  nifti_image *inputControlPointImage,
                                    nifti_image *outputControlPointImage,
                                    float4 **inputControlPointArray_d,
                                    float4 **outputControlPointArray_d);

/** Convert a displacement image into a deformation image using the image orientation header */
extern "C++"
void reg_spline_getDeformationFromDisplacement_gpu( nifti_image *image,
                                                    float4 **imageArray_d);

/** Scaling-and-squaring approach in a FFD framework */
extern "C++"
void reg_spline_scaling_squaring_gpu(   nifti_image *velocityFieldImage,
                                        nifti_image *controlPointImage,
                                        float4 **velocityArray_d,
                                        float4 **controlPointArray_d);
#endif
