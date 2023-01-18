#pragma once

#include "Content.h"

class F3dContent: public virtual Content {
public:
    F3dContent() = delete;
    F3dContent(nifti_image *referenceIn,
               nifti_image *floatingIn,
               nifti_image *controlPointGridIn,
               nifti_image *localWeightSimIn = nullptr,
               int *referenceMaskIn = nullptr,
               mat44 *transformationMatrixIn = nullptr,
               size_t bytesIn = sizeof(float));
    virtual ~F3dContent();

    // Getters
    virtual nifti_image* GetControlPointGrid() { return controlPointGrid; }
    virtual nifti_image* GetLocalWeightSim() { return localWeightSim; }
    virtual nifti_image* GetTransformationGradient() { return transformationGradient; }
    virtual nifti_image* GetVoxelBasedMeasureGradient() { return voxelBasedMeasureGradient; }
    virtual nifti_image* GetWarpedGradient() { return warpedGradient; }

    // Methods for transferring data from nifti to device
    virtual void UpdateControlPointGrid() {}
    virtual void UpdateTransformationGradient() {}
    virtual void UpdateVoxelBasedMeasureGradient() {}
    virtual void UpdateWarpedGradient() {}

    // Auxiliary methods
    virtual void ZeroTransformationGradient();
    virtual void ZeroVoxelBasedMeasureGradient();

protected:
    nifti_image *controlPointGrid = nullptr;
    nifti_image *localWeightSim = nullptr;
    nifti_image *transformationGradient = nullptr;
    nifti_image *voxelBasedMeasureGradient = nullptr;
    nifti_image *warpedGradient = nullptr;

private:
    void AllocateLocalWeightSim(nifti_image*);
    void DeallocateLocalWeightSim();
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
};