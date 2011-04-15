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

#include "_reg_bspline.h"

/* *************************************************************** */
/** reg_spline_cppComposition(nifti_image* img1, nifti_image* img2, bool type)
  * This function compose the a first control point image with a second one:
  * T(x)=Grid1(Grid2(x)).
  * Grid1 can be either displacement(disp=true) or deformation(disp=false).
  * Cubic B-Spline can be used (bspline=true) or Cubic Spline (bspline=false)
 **/
extern "C++"
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool disp,
                              bool bspline
                              );
/* *************************************************************** */
/** reg_bspline_GetJacobianMapFromVelocityField(nifti_image *img1, nifti_image *img2)
  * This function computed a Jacobian determinant map by integrating the velocity field
 **/
extern "C++"
int reg_bspline_GetJacobianMapFromVelocityField(nifti_image* velocityFieldImage,
                                                nifti_image* jacobianImage
                                                );
/* *************************************************************** */
/** reg_bspline_GetJacobianValueFromVelocityField(nifti_image *img1, nifti_image *img2, bool approx)
  * This function compute a weight based on the Jacobian determinant map.
  * The jacobian map is computed using the reg_bspline_GetJacobianMapFromVelocityField function
  * The result image header is used to know the image dimensions.
  * The Jacobian map can be computed either on all voxels or only at the control point position,
  * this is defined by the approx flag
 **/
extern "C++"
double reg_bspline_GetJacobianValueFromVelocityField(nifti_image* velocityFieldImage,
                                                     nifti_image* resultImage,
                                                     bool approx
                                                     );
/* *************************************************************** */
/** reg_bspline_GetJacobianGradientFromVelocityField(nifti_image *img1, nifti_image *img2);
  * The gradient of the Jacobian determinant weight is computing using the integration of
  * a velocity field.
  * The gradient is weighted by the weight variable and is added to the gradient image values.
  * The Jacobian determinant geradient can be computed either on all voxels or
  * only at the control point position, this is defined by the approx flag
 **/
extern "C++"
void reg_bspline_GetJacobianGradientFromVelocityField(nifti_image* velocityFieldImage,
                                                      nifti_image* resultImage,
                                                      nifti_image* gradientImage,
                                                      float weight,
                                                      bool approx
                                                      );
/* *************************************************************** */
/** reg_getControlPointPositionFromVelocityGrid(nifti_image *img1, nifti_image *img2);
  * The deformation of the control point grid (img2) is computed by integrating
  * a velocity field (img1).
 **/
extern "C++"
void reg_getControlPointPositionFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                 nifti_image *controlPointGrid);
/* *************************************************************** */
/** reg_getDeformationFieldFromVelocityGrid(nifti_image *img1, nifti_image *img2, int *mask);
  * The deformation field (img2) is computed by integrating a velocity field (img1).
  * Only the voxel within the mask will be considered. If Mask is set to NULL then
  * all the voxels will be included within the mask.
 **/
extern "C++"
void reg_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *deformationFieldImage,
                                             int *currentMask);
/* *************************************************************** */
/** reg_getPositionFromDisplacement(nifti_image *img1, nifti_image *img2)
  * This function converts a grid of control point positions into coefficient.
  * The coefficients can be used then using Cubic B-Spline to perform true interpolation
  * Using the img2 output with Cubic B-Spline basis functions is equivalent
  * to using img1 with Cubic Spline basis functions.
**/
extern "C++"
int reg_spline_Interpolant2Interpolator(nifti_image *inputImage,
                                        nifti_image *outputImage
                                        );
/* *************************************************************** */

#endif
