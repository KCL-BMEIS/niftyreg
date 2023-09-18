/*
 *  _reg_spline_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_tools_gpu.h"

/* *************************************************************** */
void reg_spline_getDeformationField_gpu(const nifti_image *controlPointImage,
                                        const nifti_image *referenceImage,
                                        const float4 *controlPointImageCuda,
                                        float4 *deformationFieldCuda,
                                        const int *maskCuda,
                                        const size_t activeVoxelNumber,
                                        const bool bspline);
/* *************************************************************** */
float reg_spline_approxBendingEnergy_gpu(const nifti_image *controlPointImage,
                                         const float4 *controlPointImageCuda);
/* *************************************************************** */
void reg_spline_approxBendingEnergyGradient_gpu(const nifti_image *controlPointImage,
                                                const float4 *controlPointImageCuda,
                                                float4 *transGradientCuda,
                                                float bendingEnergyWeight);
/* *************************************************************** */
double reg_spline_getJacobianPenaltyTerm_gpu(const nifti_image *referenceImage,
                                             const nifti_image *controlPointImage,
                                             const float4 *controlPointImageCuda,
                                             const bool approx);
/* *************************************************************** */
void reg_spline_getJacobianPenaltyTermGradient_gpu(const nifti_image *referenceImage,
                                                   const nifti_image *controlPointImage,
                                                   const float4 *controlPointImageCuda,
                                                   float4 *transGradientCuda,
                                                   const float jacobianWeight,
                                                   const bool approx);
/* *************************************************************** */
double reg_spline_correctFolding_gpu(const nifti_image *referenceImage,
                                     const nifti_image *controlPointImage,
                                     float4 *controlPointImageCuda,
                                     const bool approx);
/* *************************************************************** */
void reg_defField_compose_gpu(const nifti_image *deformationField,
                              const float4 *deformationFieldCuda,
                              float4 *deformationFieldOutCuda,
                              const size_t activeVoxelNumber);
/* *************************************************************** */
void reg_spline_getDefFieldFromVelocityGrid_gpu(nifti_image *velocityFieldGrid,
                                                nifti_image *deformationField,
                                                float4 *velocityFieldGridCuda,
                                                float4 *deformationFieldCuda,
                                                const bool updateStepNumber);
/* *************************************************************** */
void reg_defField_getJacobianMatrix_gpu(const nifti_image *deformationField,
                                        const float4 *deformationFieldCuda,
                                        float *jacobianMatricesCuda);
/* *************************************************************** */
void reg_spline_approxLinearEnergyGradient_gpu(const nifti_image *controlPointGrid,
                                               const float4 *controlPointGridCuda,
                                               float4 *transGradCuda,
                                               const float weight);
/* *************************************************************** */
