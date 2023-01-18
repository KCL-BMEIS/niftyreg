/**
 * @file _reg_f3d2.h
 * @author Marc Modat
 * @date 19/11/2011
 *
 *  Copyright (c) 2011-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_f3d.h"

/// @brief Fast Free Form Diffeomorphic Deformation registration class
template <class T>
class reg_f3d2: public reg_f3d<T> {
protected:
    nifti_image *floatingMaskImage;
    int **floatingMaskPyramid;
    int *floatingMask;
    int *backwardActiveVoxelNumber;

    nifti_image *backwardControlPointGrid;
    nifti_image *backwardDeformationFieldImage;
    nifti_image *backwardWarped;
    nifti_image *backwardWarpedGradientImage;
    nifti_image *backwardVoxelBasedMeasureGradientImage;
    nifti_image *backwardTransformationGradient;

    mat33 *backwardJacobianMatrix;

    T inverseConsistencyWeight;
    double currentIC;
    double bestIC;

    bool bchUpdate;
    bool useGradientCumulativeExp;
    int bchUpdateValue;

    // Optimiser-related function
    virtual void SetOptimiser() override;

    virtual void AllocateWarped();
    virtual void DeallocateWarped();
    virtual void AllocateDeformationField();
    virtual void DeallocateDeformationField();
    virtual void AllocateWarpedGradient();
    virtual void DeallocateWarpedGradient();
    virtual void AllocateVoxelBasedMeasureGradient();
    virtual void DeallocateVoxelBasedMeasureGradient();
    virtual void AllocateTransformationGradient();
    virtual void DeallocateTransformationGradient();
    virtual void DeallocateCurrentInputImage();

    virtual double ComputeBendingEnergyPenaltyTerm() override;
    virtual double ComputeLinearEnergyPenaltyTerm() override;
    virtual double ComputeJacobianBasedPenaltyTerm(int) override;
    virtual double ComputeLandmarkDistancePenaltyTerm() override;
    virtual void GetDeformationField() override;
    virtual void WarpFloatingImage(int) override;
    virtual void GetVoxelBasedGradient() override;
    virtual void GetSimilarityMeasureGradient() override;
    virtual void GetObjectiveFunctionGradient() override;
    virtual void GetBendingEnergyGradient() override;
    virtual void GetLinearEnergyGradient() override;
    virtual void GetJacobianBasedGradient() override;
    virtual void GetLandmarkDistanceGradient() override;
    virtual void SetGradientImageToZero() override;
    virtual T NormaliseGradient() override;
    virtual void SmoothGradient() override;
    virtual void GetApproximatedGradient() override;
    virtual void DisplayCurrentLevelParameters() override;
    virtual void PrintInitialObjFunctionValue() override;
    virtual void PrintCurrentObjFunctionValue(T) override;
    virtual void UpdateBestObjFunctionValue() override;
    virtual double GetObjectiveFunctionValue() override;

    virtual T InitialiseCurrentLevel() override;
    virtual void UpdateParameters(float) override;
    virtual void InitialiseSimilarity() override;

    virtual void GetInverseConsistencyErrorField(bool forceAll);
    virtual double GetInverseConsistencyPenaltyTerm();
    virtual void GetInverseConsistencyGradient();
    virtual void ExponentiateGradient();

public:
    reg_f3d2(int refTimePoint, int floTimePoint);
    virtual ~reg_f3d2();

    virtual void SetFloatingMask(nifti_image*) override;
    virtual void SetInverseConsistencyWeight(T) override;
    virtual void CheckParameters() override;
    virtual void Initialise() override;
    virtual nifti_image** GetWarpedImage() override;
    virtual nifti_image* GetBackwardControlPointPositionImage() override;
    virtual bool GetSymmetricStatus() { return true; }

    virtual void UseBCHUpdate(int) override;
    virtual void UseGradientCumulativeExp() override;
    virtual void DoNotUseGradientCumulativeExp() override;
};
