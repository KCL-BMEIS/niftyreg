/**
 * @file _reg_f3d.h
 * @author Marc Modat
 * @date 19/11/2010
 *
 *  Copyright (c) 2010-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_base.h"

/// @brief Fast Free Form Deformation registration class
template <class T>
class reg_f3d: public reg_base<T> {
protected:
    NiftiImage inputControlPointGrid;
    NiftiImage controlPointGrid;
    T bendingEnergyWeight;
    T linearEnergyWeight;
    T jacobianLogWeight;
    bool jacobianLogApproximation;
    T spacing[3];
    bool gridRefinement;
    double currentWJac;
    double currentWBE;
    double currentWLE;
    double bestWJac;
    double bestWBE;
    double bestWLE;

    void InitContent(nifti_image*, nifti_image*, int*);
    virtual T InitCurrentLevel(int) override;
    virtual void DeinitCurrentLevel(int) override;
    virtual T NormaliseGradient() override;
    virtual void SmoothGradient() override;
    virtual void GetObjectiveFunctionGradient() override;
    virtual void GetApproximatedGradient() override;
    virtual void GetSimilarityMeasureGradient() override;
    virtual void GetDeformationField() override;
    virtual void DisplayCurrentLevelParameters(int) override;
    virtual double GetObjectiveFunctionValue() override;
    virtual void UpdateBestObjFunctionValue() override;
    virtual void UpdateParameters(float) override;
    virtual void SetOptimiser() override;
    virtual void PrintInitialObjFunctionValue() override;
    virtual void PrintCurrentObjFunctionValue(T) override;
    virtual void CorrectTransformation() override;
    virtual void CheckParameters() override;
    virtual void Initialise() override;

    virtual double ComputeBendingEnergyPenaltyTerm();
    virtual double ComputeLinearEnergyPenaltyTerm();
    virtual double ComputeJacobianBasedPenaltyTerm(int);
    virtual double ComputeLandmarkDistancePenaltyTerm();
    virtual void GetBendingEnergyGradient();
    virtual void GetLinearEnergyGradient();
    virtual void GetJacobianBasedGradient();
    virtual void GetLandmarkDistanceGradient();

public:
    reg_f3d(int refTimePoints, int floTimePoints);

    virtual NiftiImage GetControlPointPositionImage();
    virtual vector<NiftiImage> GetWarpedImage() override;

    virtual void SetControlPointGridImage(NiftiImage);
    virtual void SetBendingEnergyWeight(T);
    virtual void SetLinearEnergyWeight(T);
    virtual void SetJacobianLogWeight(T);
    virtual void ApproximateJacobianLog();
    virtual void DoNotApproximateJacobianLog();
    virtual void SetSpacing(unsigned, T);
    virtual void NoGridRefinement() { gridRefinement = false; }

    // F3D2 specific options
    virtual NiftiImage GetBackwardControlPointPositionImage() { return {}; }
    virtual void UseBCHUpdate(int) {}
    virtual void UseGradientCumulativeExp() {}
    virtual void DoNotUseGradientCumulativeExp() {}
    virtual void SetFloatingMask(NiftiImage) {}
    virtual void SetInverseConsistencyWeight(T) {}
};
