/**
 * @file _reg_localTrans_jac.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 23/12/2015
 *
 * Copyright (c) 2015, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANS_JAC_H
#define _REG_TRANS_JAC_H

#include "_reg_localTrans.h"

/* *************************************************************** */
/** @brief Compute the Jacobian determinant map using a cubic b-spline
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param jacobianImage Image that will be populated with the determinant
 * of the Jacobian matrix of the transformation at every voxel posision.
 */
extern "C++"
void reg_spline_GetJacobianMap(nifti_image *controlPointGridImage,
                               nifti_image *jacobianImage
                               );
/* *************************************************************** */
/** @brief Compute the average Jacobian determinant
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param referenceImage Image that defines the space of the deformation
 * field for the transformation
 * @param approx Approximate the average Jacobian determinant by using
 * only the information from the control point if the value is set to true;
 * all voxels are considered if the value is set to false.
 */
extern "C++"
double reg_spline_getJacobianPenaltyTerm(nifti_image *controlPointGridImage,
                                         nifti_image *referenceImage,
                                         bool approx,
                                         bool useHeaderInformation=false
      );
/* *************************************************************** */
/** @brief Compute the gradient at every control point position of the
 * Jacobian determinant based penalty term
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param referenceImage Image that defines the space of the deformation
 * field for the transformation
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the Jacobian determinant based penalty term is added
 * to the current values
 * @param weight The gradient of the Euclidean displacement of the control
 * point position is weighted by this value
 * @param approx Approximate the gradient by using only the information
 * from the control point if the value is set to true; all voxels are
 * considered if the value is set to false.
 */
extern "C++"
void reg_spline_getJacobianPenaltyTermGradient(nifti_image *controlPointGridImage,
                                               nifti_image *referenceImage,
                                               nifti_image *gradientImage,
                                               float weight,
                                               bool approx,
                                               bool useHeaderInformation=false
      );
/* *************************************************************** */
/** @brief Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation. This function does require
 * the control point grid to perfectly overlay the reference image.
 * @param referenceImage Image that defines the space of the deformation
 * field
 * @param controlPointGridImage Control point grid position that defines
 * the cubic B-Spline parametrisation
 * @param jacobianImage Array that is filled with the Jacobian matrices
 * for every voxel.
 */
extern "C++"
void reg_spline_GetJacobianMatrix(nifti_image *referenceImage,
                                  nifti_image *controlPointGridImage,
                                  mat33 *jacobianImage
                                  );
/* *************************************************************** */
/** @brief Correct the folding in the transformation parametrised through
 * cubic B-Spline
 * @param controlPointGridImage Image that contains the cubic B-Spline
 * parametrisation
 * @param referenceImage Image that defines the space of the transformation
 * @param approx The function can be run be considering only the control
 * point position (approx==false) or every voxel (approx==true)
 */
extern "C++"
double reg_spline_correctFolding(nifti_image *controlPointGridImage,
                                 nifti_image *referenceImage,
                                 bool approx
                                 );
/* *************************************************************** */
/** @brief Compute the Jacobian determinant at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 * @param deformationField Image that contains a deformation field
 * @param jacobianImage This image will be fill with the Jacobian
 * determinant of the transformation of every voxel.
 */
extern "C++"
void reg_defField_getJacobianMap(nifti_image *deformationField,
                                 nifti_image *jacobianImage);
/* *************************************************************** */
/** @brief Compute the Jacobian matrix at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 * @param deformationField Image that contains a deformation field
 * @param jacobianMatrices This array will be fill with the Jacobian
 * matrices of the transformation of every voxel.
 */
extern "C++"
void reg_defField_getJacobianMatrix(nifti_image *deformationField,
                                    mat33 *jacobianMatrices);
/* *************************************************************** */
/** @brief This function computed Jacobian matrices by integrating
 * the velocity field
 * @param referenceImage Image that defines the space of the deformation
 * field
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 * @param jacobianMatrices Array of matrices that will be filled with
 * the Jacobian matrices of the transformation
 */
extern "C++"
int reg_defField_GetJacobianMatFromFlowField(mat33* jacobianMatrices,
                                             nifti_image *flowFieldImage);
extern "C++"
int reg_spline_GetJacobianMatFromVelocityGrid(mat33* jacobianMatrices,
                                              nifti_image *velocityGridImage,
                                              nifti_image *referenceImage
                                              );
/* *************************************************************** */
/** @brief This function computed a Jacobian determinant map by integrating
 * the velocity grid
 * @param jacobianDetImage This image will be filled with the Jacobian
 * determinants of every voxel.
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 */
extern "C++"
int reg_defField_GetJacobianDetFromFlowField(nifti_image *jacobianDetImage,
                                             nifti_image *flowFieldImage
                                             );
extern "C++"
int reg_spline_GetJacobianDetFromVelocityGrid(nifti_image *jacobianDetImage,
                                              nifti_image *velocityGridImage);
/* *************************************************************** */


#endif
