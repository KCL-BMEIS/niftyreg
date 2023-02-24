#pragma once

#include "Content.h"
#include "_reg_optimiser.h"

class Compute {
public:
    Compute() = delete;
    Compute(Content& conIn): con(conIn) {}

    virtual void ResampleImage(int inter, float paddingValue);
    virtual double GetJacobianPenaltyTerm(bool approx);
    virtual void JacobianPenaltyTermGradient(float weight, bool approx);
    virtual double CorrectFolding(bool approx);
    virtual double ApproxBendingEnergy();
    virtual void ApproxBendingEnergyGradient(float weight);
    virtual double ApproxLinearEnergy();
    virtual void ApproxLinearEnergyGradient(float weight);
    virtual double GetLandmarkDistance(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating);
    virtual void LandmarkDistanceGradient(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating, float weight);
    virtual void GetDeformationField(bool composition, bool bspline);
    virtual void UpdateControlPointPosition(float *currentDOF, float *bestDOF, float *gradient, float scale, bool optimiseX, bool optimiseY, bool optimiseZ);
    virtual void GetImageGradient(int interpolation, float paddingValue, int activeTimepoint);
    virtual double GetMaximalLength(size_t nodeNumber, bool optimiseX, bool optimiseY, bool optimiseZ);
    virtual void NormaliseGradient(size_t nodeNumber, double maxGradLength, bool optimiseX, bool optimiseY, bool optimiseZ);
    virtual void SmoothGradient(float sigma);
    virtual void GetApproximatedGradient(InterfaceOptimiser& opt);
    virtual void GetDefFieldFromVelocityGrid(bool updateStepNumber);
    virtual void ConvolveVoxelBasedMeasureGradient(float weight);
    virtual void ExponentiateGradient(Content& conBw);
    virtual void UpdateVelocityField(float scale, bool optimiseX, bool optimiseY, bool optimiseZ);
    virtual void BchUpdate(float scale, int bchUpdateValue);
    virtual void SymmetriseVelocityFields(Content& conBw);

protected:
    Content& con;

    void ConvolveImage(nifti_image*);

private:
    template<typename Type> void GetApproximatedGradient(InterfaceOptimiser&);
    nifti_image* ScaleGradient(const nifti_image&, float);
};
