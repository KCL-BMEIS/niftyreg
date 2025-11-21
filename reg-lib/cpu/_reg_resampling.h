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

#pragma once

#include "RNifti.h"

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
 * from zero are being considered. If nullptr, all voxels are considered
 * @param interpolation Interpolation type. 0, 1 or 3 correspond to nearest neighbor, linear or cubic
 * interpolation
 * @param paddingValue Value to be used for padding when the correspondences are outside of the
 * reference image space.
 * @param dtIndicies Array of 6 integers that correspond to the "time" indicies of the diffusion tensor
 * components in the order xx,yy,zz,xy,xz,yz. If there are no DT images, pass an array of -1's
 */
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       const nifti_image *deformationField,
                       const int *mask,
                       const int interpolation,
                       const float paddingValue,
                       const bool *dtiTimePoint = nullptr,
                       const mat33 *jacMat = nullptr);
/* *************************************************************** */
void reg_resampleImage_PSF(const nifti_image *floatingImage,
                           nifti_image *warpedImage,
                           const nifti_image *deformationField,
                           const int *mask,
                           const int interpolation,
                           const float paddingValue,
                           const mat33 *jacMat,
                           const char algorithm);
/* *************************************************************** */
void reg_resampleGradient(const nifti_image *gradientImage,
                          nifti_image *warpedGradient,
                          const nifti_image *deformationField,
                          const int interpolation,
                          const float paddingValue);
/* *************************************************************** */
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warpedGradient,
                          const nifti_image *deformationField,
                          const int *mask,
                          const int interpolation,
                          const float paddingValue,
                          const int activeTimePoint,
                          const bool *dtiTimePoint = nullptr,
                          const mat33 *jacMat = nullptr,
                          const nifti_image *warpedImage = nullptr);
/* *************************************************************** */
void reg_getImageGradient_symDiff(const nifti_image *img,
                                  nifti_image *gradImg,
                                  const int *mask,
                                  const float paddingValue,
                                  const int timePoint);
/* *************************************************************** */
nifti_image* reg_makeIsotropic(nifti_image*, int);
/* *************************************************************** */
