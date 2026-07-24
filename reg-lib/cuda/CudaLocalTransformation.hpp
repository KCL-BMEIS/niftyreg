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
/** @brief Reusable device scratch for the velocity-field exponentiation.
 *
 * The scaling-and-squaring exponentiation runs on every objective-function evaluation (forward and
 * backward, plus every line-search probe) and every gradient step, each time allocating and freeing
 * the flow-field buffer and an identity mask (and, for the gradient path, the intermediate fields).
 * Passing a persistent workspace removes that per-call cudaMalloc/cudaFree churn (a large host-side
 * cost in profiling). The identity mask is a constant [0, 1, 2, ...] sequence, so it is filled once
 * and reused. Buffers only ever grow.
 */
struct ExponentiationWorkspace {
public:
    /// Identity mask [0, 1, ...voxelNumber); (re)filled only when it must grow.
    const int* GetIdentityMask(const size_t voxelNumber);
    /// Flow-field scratch of at least voxelNumber float4 (grows only).
    float4* GetFlowField(const size_t voxelNumber);
    /// Ensure `count` intermediate buffers each holding at least voxelNumber float4 (grows only).
    vector<thrust::device_vector<float4>>& GetIntermediates(const size_t count, const size_t voxelNumber);

private:
    thrust::device_vector<int> mask;                       // identity mask (filled once per size)
    thrust::device_vector<float4> flowField;               // flow-field scratch
    vector<thrust::device_vector<float4>> intermediates;   // gradient path: squaringNumber+1 fields
};
/* *************************************************************** */
void GetDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                 nifti_image *deformationField,
                                 float4 *velocityFieldGridCuda,
                                 float4 *deformationFieldCuda,
                                 const bool updateStepNumber,
                                 ExponentiationWorkspace *workspace = nullptr);
/* *************************************************************** */
void GetIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                        float4 *velocityFieldGridCuda,
                                        vector<NiftiImage>& deformationFields,
                                        vector<thrust::device_vector<float4>>& deformationFieldCudaVecs,
                                        ExponentiationWorkspace *workspace = nullptr);
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
