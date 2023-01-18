#pragma once

#include "Content.h"

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
    virtual void VoxelCentricToNodeCentric(float weight);
    virtual double GetMaximalLength(size_t nodeNumber, bool optimiseX, bool optimiseY, bool optimiseZ);
    virtual void NormaliseGradient(size_t nodeNumber, double maxGradLength);

protected:
    Content& con;
};
