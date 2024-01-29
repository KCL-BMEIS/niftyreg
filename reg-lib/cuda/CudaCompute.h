#pragma once

#include "Compute.h"
#include "CudaCommon.hpp"

class CudaCompute: public Compute {
public:
    CudaCompute(Content& con): Compute(con) {}

    virtual void ResampleImage(int interpolation, float paddingValue) override;
    virtual double GetJacobianPenaltyTerm(bool approx) override;
    virtual void JacobianPenaltyTermGradient(float weight, bool approx) override;
    virtual double CorrectFolding(bool approx) override;
    virtual double ApproxBendingEnergy() override;
    virtual void ApproxBendingEnergyGradient(float weight) override;
    virtual double ApproxLinearEnergy() override;
    virtual void ApproxLinearEnergyGradient(float weight) override;
    virtual double GetLandmarkDistance(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating) override;
    virtual void LandmarkDistanceGradient(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating, float weight) override;
    virtual void GetDeformationField(bool composition, bool bspline) override;
    virtual void UpdateControlPointPosition(float *currentDof, const float *bestDof, const float *gradient, const float scale, const bool optimiseX, const bool optimiseY, const bool optimiseZ) override;
    virtual void GetImageGradient(int interpolation, float paddingValue, int activeTimePoint) override;
    virtual double GetMaximalLength(bool optimiseX, bool optimiseY, bool optimiseZ) override;
    virtual void NormaliseGradient(double maxGradLength, bool optimiseX, bool optimiseY, bool optimiseZ) override;
    virtual void SmoothGradient(float sigma) override;
    virtual void GetApproximatedGradient(InterfaceOptimiser& opt) override;
    virtual void GetDefFieldFromVelocityGrid(const bool updateStepNumber) override;
    virtual void ConvolveVoxelBasedMeasureGradient(float weight) override;
    virtual void ExponentiateGradient(Content& conBw) override;
    virtual void UpdateVelocityField(float scale, bool optimiseX, bool optimiseY, bool optimiseZ) override;
    virtual void BchUpdate(float scale, int bchUpdateValue) override;
    virtual void SymmetriseVelocityFields(Content& conBw) override;
    virtual void GetAffineDeformationField(bool compose) override;

#ifndef NR_TESTING
protected:
#endif
    virtual void DefFieldCompose(const nifti_image *defField) override;
    virtual NiftiImage ResampleGradient(int interpolation, float padding) override;
    virtual void VoxelCentricToNodeCentric(float weight) override;

private:
    void ConvolveImage(const nifti_image*, float4*);
    Cuda::UniquePtr<float4> ScaleGradient(const float4*, const size_t, const float);
};
