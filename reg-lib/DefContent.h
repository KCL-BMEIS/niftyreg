#pragma once

#include "Content.h"

class DefContent: public virtual Content {
public:
    DefContent() = delete;
    DefContent(NiftiImage& referenceIn,
               NiftiImage& floatingIn,
               NiftiImage *localWeightSimIn = nullptr,
               int *referenceMaskIn = nullptr,
               mat44 *transformationMatrixIn = nullptr,
               size_t bytesIn = sizeof(float));

    // Getters
    virtual NiftiImage& GetLocalWeightSim() { return localWeightSim; }
    virtual NiftiImage& GetVoxelBasedMeasureGradient() { return voxelBasedMeasureGradient; }
    virtual NiftiImage& GetWarpedGradient() { return warpedGradient; }

    // Methods for transferring data from nifti to device
    virtual void UpdateVoxelBasedMeasureGradient() {}
    virtual void UpdateWarpedGradient() {}

    // Auxiliary methods
    virtual void ZeroVoxelBasedMeasureGradient();

protected:
    NiftiImage localWeightSim;
    NiftiImage voxelBasedMeasureGradient;
    NiftiImage warpedGradient;

private:
    void AllocateLocalWeightSim(NiftiImage&);
};
