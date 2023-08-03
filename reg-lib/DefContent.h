#pragma once

#include "Content.h"

class DefContent: public virtual Content {
public:
    DefContent() = delete;
    DefContent(nifti_image *referenceIn,
               nifti_image *floatingIn,
               nifti_image *localWeightSimIn = nullptr,
               int *referenceMaskIn = nullptr,
               mat44 *transformationMatrixIn = nullptr,
               size_t bytesIn = sizeof(float));
    virtual ~DefContent();

    // Getters
    virtual nifti_image* GetLocalWeightSim() { return localWeightSim; }
    virtual nifti_image* GetVoxelBasedMeasureGradient() { return voxelBasedMeasureGradient; }
    virtual nifti_image* GetWarpedGradient() { return warpedGradient; }

    // Methods for transferring data from nifti to device
    virtual void UpdateVoxelBasedMeasureGradient() {}
    virtual void UpdateWarpedGradient() {}

    // Auxiliary methods
    virtual void ZeroVoxelBasedMeasureGradient();

protected:
    nifti_image *localWeightSim = nullptr;
    nifti_image *voxelBasedMeasureGradient = nullptr;
    nifti_image *warpedGradient = nullptr;

private:
    void AllocateLocalWeightSim(nifti_image*);
    void DeallocateLocalWeightSim();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
};
