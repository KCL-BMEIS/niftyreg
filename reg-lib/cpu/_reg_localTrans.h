/**
 * @file _reg_localTrans.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 25/03/2009
 *
 * Created by Marc Modat on 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 * The reg_defFieldInvert function has been initially written by
 * Marcel van Herk (CMIC / NKI / AVL)
 */

#ifndef _REG_TRANS_H
#define _REG_TRANS_H

#include "float.h"
#include "_reg_globalTrans.h"
#include "_reg_splineBasis.h"

/* *********************************************** */
/* ****      CUBIC SPLINE BASED FUNCTIONS     **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief Generate a control point grid image based on the dimension of a
 * reference image and on a spacing.
 * The function set the qform and sform code to overlay the reference
 * image.
 * @param controlPointGridImage The resulting control point grid will be
 * store in this pointer
 * @param referenceImage Reference image which dimension will be used to
 * define the control point grid image space
 * @param spacingMillimeter Control point spacing along each axis
 */
extern "C++" template <class DTYPE>
void reg_createControlPointGrid(nifti_image **controlPointGridImage,
                                nifti_image *referenceImage,
                                float *spacingMillimeter);

extern "C++" template <class DTYPE>
void reg_createSymmetricControlPointGrids(nifti_image **forwardGridImage,
                                          nifti_image **backwardGridImage,
                                          nifti_image *referenceImage,
                                          nifti_image *floatingImage,
                                          mat44 *forwardAffineTrans,
                                          float *spacing);

/* *************************************************************** */
/** @brief Compute a dense deformation field in the space of a reference
 * image from a grid of control point.
 * @param controlPointGridImage Control point grid that contains the deformation
 * parametrisation
 * @param deformationField Output image that will be populated with the deformation field
 * @param mask Array that contains the a mask. Any voxel with a positive value is included
 * into the mask
 * @param composition A composition scheme is used if this value is set to true,
 * the deformation is starting from a blank grid otherwise.
 * @param bspline A cubic B-Spline scheme is used if the value is set to true,
 * a cubic spline scheme is used otherwise (interpolant spline).
 */
extern "C++"
void reg_spline_getDeformationField(nifti_image *controlPointGridImage,
                                    nifti_image *deformationField,
                                    int *mask = NULL,
                                    bool composition = false,
                                    bool bspline = true,
                                    bool force_no_lut = false);
/* *************************************************************** */
/** @brief Upsample an image from voxel space to node space using
 * millimiter correspendences.
 * @param nodeImage This image is a coarse representation of the
 * transformation (typically a grid of control point). This image
 * values are going to be updated
 * @param voxelImage This image contains a dense representation
 * if the transformation (typically a voxel-based gradient)
 * @param weight The values from used to update the node image
 * will be multiplied by the weight
 * @param update The values in node image will be incremented if
 * update is set to true; a blank node image is considered otherwise
 */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight,
                                  bool update,
                                  mat44 *voxelToMillimeter = NULL
      );
/* *************************************************************** */
/** @brief Refine a grid of control points
 * @param referenceImage Image that defined the space of the reference
 * image
 * @param controlPointGridImage This control point grid will be refined
 * by dividing the control point spacing by a ratio of 2
 */
extern "C++"
void reg_spline_refineControlPointGrid(nifti_image *controlPointGridImage,
                                       nifti_image *referenceImage = NULL
      );
/* *************************************************************** */
/** @brief This function compose the a first control point image with a second one:
 * Grid2(x) <= Grid1(Grid2(x)).
 * Grid1 and Grid2 have to contain either displacement or deformation.
 * The output will be a deformation field if grid1 is a deformation,
 * The output will be a displacement field if grid1 is a displacement.
 * @param grid1 Image that contains the first grid of control points
 * @param grid2 Image that contains the second grid of control points
 * @param displacement1 The first grid is a displacement field if this
 * value is set to true, a deformation field otherwise
 * @param displacement2 The second grid is a displacement field if this
 * value is set to true, a deformation field otherwise
 * @param Cubic B-Spline can be used (bspline==true)
 * or cubic Spline (bspline==false)
 */
extern "C++"
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline
                              );
/* *************************************************************** */
/** @brief Preforms the composition of two deformation fields
 * The deformation field image is applied to the second image:
 * dfToUpdate. Both images are expected to contain deformation
 * field.
 * @param deformationField Image that contains the deformation field
 * that will be applied
 * @param dfToUpdate Image that contains the deformation field that
 * is being updated
 * @param mask Mask overlaid on the dfToUpdate field where only voxel
 * within the mask will be updated. All positive values in the maks
 * are considered as belonging to the mask.
 */
extern "C++"
void reg_defField_compose(nifti_image *deformationField,
                          nifti_image *dfToUpdate,
                          int *mask);
/* *************************************************************** */
/** @brief Compute the inverse of a deformation field
 * @author Marcel van Herk (CMIC / NKI / AVL)
 * @param inputDeformationField Image that contains the deformation
 * field to invert.
 * @param outputDeformationField Image that will contains the inverse
 * of the input deformation field
 * @param tolerance Tolerance value for the optimisation. Set to nan
 * for the default value.
 */
extern "C++"
void reg_defFieldInvert(nifti_image *inputDeformationField,
                        nifti_image *outputDeformationField,
                        float tolerance);
/* *************************************************************** */
extern "C++"
void reg_defField_getDeformationFieldFromFlowField(nifti_image *flowFieldImage,
                                                   nifti_image *deformationFieldImage,
                                                   bool updateStepNumber);

/* *************************************************************** */
/** @brief The deformation field (img2) is computed by integrating
 * a velocity Grid (img1)
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 * @param deformationFieldImage Deformation field image that will
 * be filled using the exponentiation of the velocity field.
 */
extern "C++"
void reg_spline_getDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                            nifti_image *deformationFieldImage,
                                            bool updateStepNumber);
/* *************************************************************** */
extern "C++"
void reg_spline_getIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                                   nifti_image **deformationFieldImage);
/* *************************************************************** */
extern "C++"
void reg_spline_getFlowFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *flowField);

/* *************************************************************** */


/* *********************************************** */
/* ****            OTHER FUNCTIONS            **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief This function compute the BCH update using an initial verlocity field
 * and its gradient.
 * @param img1 Image that contains the velocity field parametrisation
 * This image is updated
 * @param img2 This image contains the gradient to use
 * @param type The type encodes the number of component of the serie
 * to be considered:
 * 0 - w=u+v
 * 1 - w=u+v+0.5*[u,v]
 * 2 - w=u+v+0.5*[u,v]+[u,[u,v]]/12
 * 3 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12
 * 4 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12-[v,[u,[u,g]]]/24
 */
extern "C++"
void compute_BCH_update(nifti_image *img1,
                        nifti_image *img2,
                        int type);
/* *************************************************************** */
/** @brief This function deconvolve an image by a cubic B-Spline kernel
 * in order to get cubic B-Spline coefficient
 * @param img Image to be deconvolved
 */
extern "C++"
void reg_spline_GetDeconvolvedCoefficents(nifti_image *img);
/* *************************************************************** */
#endif
