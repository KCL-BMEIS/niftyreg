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
    nifti_image *inputControlPointGrid; // pointer to external
    nifti_image *controlPointGrid;
    T bendingEnergyWeight;
    T linearEnergyWeight;
    T jacobianLogWeight;
    bool jacobianLogApproximation;
    T spacing[3];

    // nifti_image *transformationGradient;
    bool gridRefinement;

    double currentWJac;
    double currentWBE;
    double currentWLE;
    double bestWJac;
    double bestWBE;
    double bestWLE;

    // virtual void AllocateTransformationGradient() override;
    // virtual void DeallocateTransformationGradient() override;
    virtual T InitialiseCurrentLevel(nifti_image *reference) override;

    virtual double ComputeBendingEnergyPenaltyTerm();
    virtual double ComputeLinearEnergyPenaltyTerm();
    virtual double ComputeJacobianBasedPenaltyTerm(int);
    virtual double ComputeLandmarkDistancePenaltyTerm();

    virtual void GetBendingEnergyGradient();
    virtual void GetLinearEnergyGradient();
    virtual void GetJacobianBasedGradient();
    virtual void GetLandmarkDistanceGradient();
    // virtual void SetGradientImageToZero() override;
    virtual T NormaliseGradient() override;
    virtual void SmoothGradient() override;
    virtual void GetObjectiveFunctionGradient() override;
    virtual void GetApproximatedGradient() override;
    void GetSimilarityMeasureGradient();

    virtual void GetDeformationField() override;
    virtual void DisplayCurrentLevelParameters() override;

    virtual double GetObjectiveFunctionValue() override;
    virtual void UpdateBestObjFunctionValue() override;
    virtual void UpdateParameters(float) override;
    virtual void SetOptimiser() override;

    virtual void PrintInitialObjFunctionValue() override;
    virtual void PrintCurrentObjFunctionValue(T) override;

    virtual void CorrectTransformation() override;

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void *paramsProgressCallback;

public:
    reg_f3d(int refTimePoint, int floTimePoint);
    virtual ~reg_f3d();

    void SetControlPointGridImage(nifti_image*);
    void SetBendingEnergyWeight(T);
    void SetLinearEnergyWeight(T);
    void SetJacobianLogWeight(T);
    void ApproximateJacobianLog();
    void DoNotApproximateJacobianLog();
    void SetSpacing(unsigned int, T);

    void NoGridRefinement() { gridRefinement = false; }
    // F3D2 specific options
    virtual void SetCompositionStepNumber(int) {}
    virtual void ApproximateComposition() {}
    virtual void UseSimilaritySymmetry() {}
    virtual void UseBCHUpdate(int) {}
    virtual void UseGradientCumulativeExp() {}
    virtual void DoNotUseGradientCumulativeExp() {}

    // f3d_sym specific options
    virtual void SetFloatingMask(nifti_image*) {}
    virtual void SetInverseConsistencyWeight(T) {}
    virtual nifti_image *GetBackwardControlPointPositionImage() { return nullptr; }

    // f3d_gpu specific option
    virtual int CheckMemoryMB() { return EXIT_SUCCESS; }

    virtual void CheckParameters() override;
    virtual void Initialise() override;
    virtual void InitContent(nifti_image *reference, nifti_image *floating, int *mask) override;
    virtual void DeinitContent() override;
    virtual nifti_image* GetControlPointPositionImage();
    virtual nifti_image** GetWarpedImage() override;
};
