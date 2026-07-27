#pragma once

#include "Compute.h"
#include "CudaCommon.hpp"

namespace NiftyReg::Cuda { struct ExponentiationWorkspace; struct KernelConvolutionWorkspace; }

class CudaCompute: public Compute {
public:
    // Constructor and destructor are defined in the .cu so the incomplete ExponentiationWorkspace
    // can be held by unique_ptr here without pulling thrust into this host-included header (the
    // unique_ptr's destructor, needed by both, is only instantiated where the type is complete).
    CudaCompute(Content& con);
    ~CudaCompute();

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
    virtual void DefFieldCompose(const NiftiImage& defField) override;
    virtual NiftiImage ResampleGradient(int interpolation, float padding) override;
    virtual void VoxelCentricToNodeCentric(float weight) override;

private:
    void ConvolveImage(const NiftiImage&, float4*);
    Cuda::UniquePtr<float4> ScaleGradient(const float4*, const size_t, const float);
    // Reusable scratch for the velocity-field exponentiation, kept across iterations to avoid
    // per-call cudaMalloc/cudaFree (lazily created on first -vel use).
    std::unique_ptr<NiftyReg::Cuda::ExponentiationWorkspace> expWorkspace;
    // Reusable scratch for the voxel-based-gradient smoothing convolutions (three per-axis passes
    // per iteration), so their internal buffers are pooled rather than reallocated each pass.
    std::unique_ptr<NiftyReg::Cuda::KernelConvolutionWorkspace> convWorkspace;
};
