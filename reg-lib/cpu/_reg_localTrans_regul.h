/**
 * @file _reg_localTrans_regul.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 22/12/2015
 *
 * Copyright (c) 2015, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANS_REG_H
#define _REG_TRANS_REG_H

#include "_reg_splineBasis.h"

/* *************************************************************** */
/** @brief Compute and return the average bending energy computed using cubic b-spline.
 * The value is approximated as the bending energy is computed at
 * the control point position only.
 * @param controlPointGridImage Control point grid that contains the deformation
 * parametrisation
 * @return The normalised bending energy. Normalised by the number of voxel
 */
extern "C++"
double reg_spline_approxBendingEnergy(nifti_image *controlPointGridImage);
/* *************************************************************** */
/** @brief Compute and return the approximated (at the control point position)
 * bending energy gradient for each control point
 * @param controlPointGridImage Image that contains the control point
 * grid used to parametrise the transformation
 * @param gradientImage Image of identical size that the control
 * point grid image. The gradient of the bending-energy will be added
 * at every control point position.
 * @param weight Scalar which will be multiplied by the bending-energy gradient
 */
extern "C++"
void reg_spline_approxBendingEnergyGradient(nifti_image *controlPointGridImage,
                                            nifti_image *gradientImage,
                                            float weight
                                            );
/* *************************************************************** */
/** @brief Compute and return the linear elastic energy terms.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @return The normalised linear energy. Normalised by the number of voxel
 */
extern "C++"
double reg_spline_linearEnergy(nifti_image *referenceImage,
                               nifti_image *controlPointGridImage);
/* *************************************************************** */
/** @brief Compute and return the linear elastic energy terms approximated
 * at the control point positions only.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @return The normalised linear energy. Normalised by the number of voxel
 */
extern "C++"
double reg_spline_approxLinearEnergy(nifti_image *controlPointGridImage);
/* *************************************************************** */
/** @brief Compute the gradient of the linear elastic energy terms
 * computed at all voxel position.
 * @param referenceImage Image that contains the dense space
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the linear elasticily terms are added to the
 * current values
 * @param weight Weight to apply to the term of the penalty
 */
extern "C++"
void reg_spline_linearEnergyGradient(nifti_image *referenceImage,
                                     nifti_image *controlPointGridImage,
                                     nifti_image *gradientImage,
                                     float weight
                                     );
/* *************************************************************** */
/** @brief Compute the gradient of the linear elastic energy terms
 * approximated at the control point positions only.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the linear elasticily terms are added to the
 * current values
 * @param weight Weight to apply to the term of the penalty
 */
extern "C++"
void reg_spline_approxLinearEnergyGradient(nifti_image *controlPointGridImage,
                                           nifti_image *gradientImage,
                                           float weight
                                           );
/* *************************************************************** */
/** @brief Compute and return the linear elastic energy terms.
 * @param deformationField Image that contains the transformation.
 * @return The normalised linear energy. Normalised by the number of voxel
 */
extern "C++"
double reg_defField_linearEnergy(nifti_image *deformationField);
/* *************************************************************** */
/** @brief Compute and return the linear elastic energy terms.
 * @param deformationField Image that contains the transformation.
 * @param weight Weight to apply to the term of the penalty
 */
extern "C++"
void reg_defField_linearEnergyGradient(nifti_image *deformationField,
                                       nifti_image *gradientImage,
                                       float weight);
/* *************************************************************** */
/** @Brief Compute the distance between two set of points given a
 * transformation
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param landmarkNumber Number of landmark defined in each image
 * @param landmarkReference Landmark in the reference image
 * @param landmarkFloating Landmark in the floating image
 */
extern "C++"
double reg_spline_getLandmarkDistance(nifti_image *controlPointImage,
                                      size_t landmarkNumber,
                                      float *landmarkReference,
                                      float *landmarkFloating);
/* *************************************************************** */
/** @Brief Compute the gradient of the distance between two set of
 * points given a transformation
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param gradientImage Image that contains the gradient in the space
 * of the transformation parametrisation
 * @param landmarkNumber Number of landmark defined in each image
 * @param landmarkReference Landmark in the reference image
 * @param landmarkFloating Landmark in the floating image
 * @param weight weight to apply to the gradient
 */
extern "C++"
void reg_spline_getLandmarkDistanceGradient(nifti_image *controlPointImage,
                                            nifti_image *gradientImage,
                                            size_t landmarkNumber,
                                            float *landmarkReference,
                                            float *landmarkFloating,
                                            float weight);
/* *************************************************************** */
/** @brief Compute and return a pairwise energy.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @return The normalised pariwise energy. Normalised by the number of voxel
 */
extern "C++"
void reg_spline_approxLinearPairwiseGradient(nifti_image *controlPointGridImage,
                                             nifti_image *gradientImage,
                                             float weight
                                             );
/* *************************************************************** */
extern "C++"
double reg_spline_approxLinearPairwise(nifti_image *controlPointGridImage);
/* *************************************************************** */
#endif
