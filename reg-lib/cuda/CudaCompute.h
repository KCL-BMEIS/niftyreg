#pragma once

#include "Compute.h"

class CudaCompute: public Compute {
public:
    CudaCompute(Content *con) : Compute(con) {}

    virtual void ResampleImage(int inter, float paddingValue) override;
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
    virtual void UpdateControlPointPosition(float *currentDOF, float *bestDOF, float *gradient, float scale, bool optimiseX, bool optimiseY, bool optimiseZ) override;
    virtual void GetImageGradient(int interpolation, float paddingValue, int activeTimepoint) override;
    virtual void VoxelCentricToNodeCentric(float weight) override;
    virtual double GetMaximalLength(size_t nodeNumber, bool optimiseX, bool optimiseY, bool optimiseZ) override;
    virtual void NormaliseGradient(size_t nodeNumber, double maxGradLength) override;
};
