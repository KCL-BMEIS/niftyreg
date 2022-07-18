/**
 * @file _reg_resampling.h
 * @author Marc Modat
 * @date 24/03/2009
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_H
#define _REG_RESAMPLING_H

#include "nifti1_io.h"

/** @brief This function resample a floating image into the space of a reference/warped image.
 * The deformation is provided by a 4D nifti image which is in the space of the reference image.
 * In the 4D image, for each voxel i,j,k, the position in the real word for the floating image is store.
 * Interpolation can be nearest Neighbor (0), linear (1) or cubic spline (3).
 * The cubic spline interpolation assume a padding value of 0
 * The padding value for the NN and the LIN interpolation are user defined.
 * @param floatingImage Floating image that is interpolated
 * @param warpedImage Warped image that is being generated
 * @param deformationField Vector field image that contains the dense correspondences
 * @param mask Array that contains information about the mask. Only voxel with mask value different
 * from zero are being considered. If NULL, all voxels are considered
 * @param interp Interpolation type. 0, 1 or 3 correspond to nearest neighbor, linear or cubic
 * interpolation
 * @param paddingValue Value to be used for padding when the correspondences are outside of the
 * reference image space.
 * @param dtIndicies Array of 6 integers that correspond to the "time" indicies of the diffusion tensor
 * components in the order xx,yy,zz,xy,xz,yz. If there are no DT images, pass an array of -1's
 */
extern "C++"
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       nifti_image *deformationField,
                       int *mask,
                       int interp,
                       float paddingValue,
                       bool *dti_timepoint = NULL,
                       mat33 * jacMat = NULL);
extern "C++"
void reg_resampleImage_PSF(nifti_image *floatingImage,
                           nifti_image *warpedImage,
                           nifti_image *deformationField,
                           int *mask,
                           int interp,
                           float paddingValue,
                           mat33 * jacMat,
                           char algorithm);


extern "C++"
void reg_resampleGradient(nifti_image *gradientImage,
                          nifti_image *warImgGradient,
                          nifti_image *deformationField,
                          int interp,
                          float paddingValue);

extern "C++"
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warImgGradient,
                          nifti_image *deformationField,
                          int *mask,
                          int interp,
                          float paddingValue,
                          int active_timepoint,
                          bool *dti_timepoint = NULL,
                          mat33 *jacMat = NULL,
                          nifti_image *warpedImage = NULL);

extern "C++"
void reg_getImageGradient_symDiff(nifti_image* inputImg,
                                  nifti_image* gradImg,
                                  int *mask,
                                  float padding_value,
                                  int timepoint);
extern "C++"
nifti_image *reg_makeIsotropic(nifti_image *, int);

#endif
