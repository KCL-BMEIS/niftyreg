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
    nifti_image *controlPointGridBw;
    mat44 *affineTransformationBw;
    T inverseConsistencyWeight;
    bool bchUpdate;
    bool useGradientCumulativeExp;
    int bchUpdateValue;

    // Content backwards
    F3dContent *conBw = nullptr;

    // Compute backwards
    Compute *computeBw = nullptr;

    virtual void SetOptimiser() override;
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
    virtual T NormaliseGradient() override;
    virtual void SmoothGradient() override;
    virtual void GetApproximatedGradient() override;
    virtual void DisplayCurrentLevelParameters(int) override;
    virtual void PrintInitialObjFunctionValue() override;
    virtual void PrintCurrentObjFunctionValue(T) override;
    virtual void UpdateBestObjFunctionValue() override;
    virtual double GetObjectiveFunctionValue() override;
    void InitContent(nifti_image*, nifti_image*, int*);
    virtual T InitCurrentLevel(int) override;
    virtual void DeinitCurrentLevel(int) override;
    virtual void UpdateParameters(float) override;
    virtual void InitialiseSimilarity() override;
    virtual void CheckParameters() override;
    virtual void Initialise() override;

    virtual void ExponentiateGradient();

public:
    reg_f3d2(int refTimePoint, int floTimePoint);
    virtual ~reg_f3d2();

    virtual nifti_image* GetBackwardControlPointPositionImage() override;
    virtual nifti_image** GetWarpedImage() override;
    virtual bool GetSymmetricStatus() override { return true; }

    virtual void SetFloatingMask(nifti_image*) override;
    virtual void SetInverseConsistencyWeight(T) override;
    virtual void UseBCHUpdate(int) override;
    virtual void UseGradientCumulativeExp() override;
    virtual void DoNotUseGradientCumulativeExp() override;
};
