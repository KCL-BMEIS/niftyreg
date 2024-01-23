/*
 *  CudaLocalTransformation.hpp
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

#include "CudaTools.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
void GetDeformationFromDisplacement(nifti_image *image, float4 *imageCuda);
/* *************************************************************** */
void GetDisplacementFromDeformation(nifti_image *image, float4 *imageCuda);
/* *************************************************************** */
template<bool composition, bool bspline>
void GetDeformationField(const nifti_image *controlPointImage,
                         const nifti_image *referenceImage,
                         const float4 *controlPointImageCuda,
                         float4 *deformationFieldCuda,
                         const int *maskCuda,
                         const size_t activeVoxelNumber);
/* *************************************************************** */
template<bool is3d>
double ApproxBendingEnergy(const nifti_image *controlPointImage,
                           const float4 *controlPointImageCuda);
/* *************************************************************** */
template<bool is3d>
void ApproxBendingEnergyGradient(nifti_image *controlPointImage,
                                 float4 *controlPointImageCuda,
                                 float4 *transGradientCuda,
                                 float bendingEnergyWeight);
/* *************************************************************** */
double GetJacobianPenaltyTerm(const nifti_image *referenceImage,
                              const nifti_image *controlPointImage,
                              const float4 *controlPointImageCuda,
                              const bool approx);
/* *************************************************************** */
void GetJacobianPenaltyTermGradient(const nifti_image *referenceImage,
                                    const nifti_image *controlPointImage,
                                    const float4 *controlPointImageCuda,
                                    float4 *transGradientCuda,
                                    const float jacobianWeight,
                                    const bool approx);
/* *************************************************************** */
double CorrectFolding(const nifti_image *referenceImage,
                      const nifti_image *controlPointImage,
                      float4 *controlPointImageCuda,
                      const bool approx);
/* *************************************************************** */
template<bool is3d>
void DefFieldCompose(const nifti_image *deformationField,
                     const float4 *deformationFieldCuda,
                     float4 *deformationFieldOutCuda);
/* *************************************************************** */
void GetDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                 nifti_image *deformationField,
                                 float4 *velocityFieldGridCuda,
                                 float4 *deformationFieldCuda,
                                 const bool updateStepNumber);
/* *************************************************************** */
void GetIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                        float4 *velocityFieldGridCuda,
                                        vector<NiftiImage>& deformationFields,
                                        vector<thrust::device_vector<float4>>& deformationFieldCudaVecs);
/* *************************************************************** */
void GetJacobianMatrix(const nifti_image *deformationField,
                       const float4 *deformationFieldCuda,
                       float *jacobianMatricesCuda);
/* *************************************************************** */
template<bool is3d>
double ApproxLinearEnergy(const nifti_image *controlPointGrid,
                          const float4 *controlPointGridCuda);
/* *************************************************************** */
template<bool is3d>
void ApproxLinearEnergyGradient(const nifti_image *controlPointGrid,
                                const float4 *controlPointGridCuda,
                                float4 *transGradCuda,
                                const float weight);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
