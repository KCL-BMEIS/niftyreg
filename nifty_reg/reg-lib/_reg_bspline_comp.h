/*
 *  _reg_bspline_comp.h
 *  
 *
 *  Created by Marc Modat on 25/03/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_COMP_H
#define _REG_BSPLINE_COMP_H

#define SCALING_VALUE 256
#define SQUARING_VALUE 8

#include "nifti1_io.h"
#include "_reg_affineTransformation.h"

#if _USE_SSE
    #include <emmintrin.h>
#endif


/** apply_scaling_squaring(nifti_image* img1, nifti_image* img2)
  * This function applies a squaring approach of the velocity field img1
  * in order to produce a control point position img2.
**/
extern "C++"
void reg_spline_scaling_squaring(   nifti_image *velocityFieldImage,
                                    nifti_image *controlPointImage
                                    );

/** reg_spline_cppComposition(nifti_image* img1, nifti_image* img2, bool type)
  * This function compose the a first control point image with a second one.
  * Type is set to 0 if img1 contains displacement and to 1 if it contains
  * position.
  * img2 always expect to be displacement and they will be multiplied by ratio.
**/
extern "C++"
int reg_spline_cppComposition(  nifti_image *positionGridImage,
                                nifti_image *decomposedGridImage,
                                float ratio,
                                bool type
                                );

/** reg_getDisplacementFromPosition(nifti_image *)
  * This function converts a control point grid containing positions
  * into a control point grid containing displacements.
  * The conversion is done using the appropriate qform/sform
**/
extern "C++" template<class PrecisionTYPE>
int reg_getDisplacementFromPosition(nifti_image *controlPointImage);

/** reg_getPositionFromDisplacement(nifti_image *)
  * This function converts a control point grid containing displacements
  * into a control point grid containing displacementspositions.
  * The conversion is done using the appropriate qform/sform
**/
extern "C++" template<class PrecisionTYPE>
int reg_getPositionFromDisplacement(nifti_image *controlPointImage);

/** reg_getPositionFromDisplacement(nifti_image *img1, nifti_image *img2)
  * This function converts a control point grid values into coefficients
  * in order to perform true interpolation
**/
extern "C++"
int reg_spline_Interpolant2Interpolator(nifti_image *inputImage,
                                        nifti_image *outputImage
                                        );

/** reg_bspline_GetJacobianMapFromVelocityField(nifti_image *img1, nifti_image *img2)
  * This function computed a Jacobian determinant using a squaring approach
  * applied to a velocity field
**/
extern "C++"
int reg_bspline_GetJacobianMapFromVelocityField(nifti_image* velocityFieldImage,
                                                nifti_image* jacobianImage
                                                );

/** reg_bspline_GetJacobianValueFromVelocityField(nifti_image *img1, nifti_image *img2, bool approx)
  * This function integrate all the Jacobian determinant using a squaring approach
  * applied to a velocity field. The result image header is used to know the image
  * dimensions.
**/
extern "C++"
double reg_bspline_GetJacobianValueFromVelocityField(   nifti_image* velocityFieldImage,
                                                        nifti_image* resultImage,
                                                        bool approx
                                                        );

/** reg_bspline_GetJacobianGradientFromVelocityField(nifti_image *img1, nifti_image *img2);
  * The gradient of the Jacobian-based penalty term is computed using the scaling-and-squaring
  * approach. The value can be approximated or fully computed
**/
extern "C++"
void reg_bspline_GetJacobianGradientFromVelocityField(   nifti_image* velocityFieldImage,
                                                            nifti_image* resultImage,
                                                            nifti_image* gradientImage,
                                                            float weight,
                                                            bool approx
                                                            );

/** reg_bspline_CorrectFoldingFromVelocityField(nifti_image *img1, nifti_image *img2, bool approx);
  * This function aims to removed the folded area by computing the negative Jacobian
  * determinant gradient
  * It also return the current Jacobian penalty term value.
**/
extern "C++"
double reg_bspline_CorrectFoldingFromVelocityField( nifti_image* velocityFieldImage,
                                                    nifti_image* targetImage,
                                                    bool approx
                                                    );

/** reg_bspline_CorrectFoldingFromVelocityField(nifti_image *img1, nifti_image *img2, bool approx);
  * This function aims to removed the folded area by computing the negative Jacobian
  * determinant gradient
  * It also return the current Jacobian penalty term value.
**/
extern "C++"
double reg_bspline_CorrectApproximatedFoldingFromCPP(   nifti_image* controlPointImage,
                                                        nifti_image* velocityFieldImage,
                                                        nifti_image* targetImage,
                                                        bool approx
                                                        );

#endif
