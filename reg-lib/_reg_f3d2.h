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
    NiftiImage floatingMaskImage;
    vector<unique_ptr<int[]>> floatingMaskPyramid;
    NiftiImage controlPointGridBw;
    unique_ptr<mat44> affineTransformationBw;
    T inverseConsistencyWeight;
    bool bchUpdate;
    bool useGradientCumulativeExp;
    int bchUpdateValue;

    // Content backwards
    unique_ptr<F3dContent> conBw;

    // Compute backwards
    unique_ptr<Compute> computeBw;

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
    void InitContent(nifti_image*, nifti_image*, int*, int*);
    virtual T InitCurrentLevel(int) override;
    virtual void DeinitCurrentLevel(int) override;
    virtual void UpdateParameters(float) override;
    virtual void InitialiseSimilarity() override;
    virtual void CheckParameters() override;
    virtual void Initialise() override;

    virtual void ExponentiateGradient();

public:
    reg_f3d2(int refTimePoints, int floTimePoints);

    virtual NiftiImage GetBackwardControlPointPositionImage() override;
    virtual vector<NiftiImage> GetWarpedImage() override;
    virtual bool GetSymmetricStatus() override { return true; }

    virtual void SetFloatingMask(NiftiImage) override;
    virtual void SetInverseConsistencyWeight(T) override;
    virtual void UseBCHUpdate(int) override;
    virtual void UseGradientCumulativeExp() override;
    virtual void DoNotUseGradientCumulativeExp() override;
};
