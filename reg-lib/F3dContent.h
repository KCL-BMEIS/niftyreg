#pragma once

#include "Content.h"

class F3dContent: public virtual Content {
public:
    F3dContent() = delete;
    F3dContent(nifti_image *referenceIn,
               nifti_image *floatingIn,
               nifti_image *controlPointGridIn,
               nifti_image *localWeightSimIn,
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

    // Setters
    virtual void SetControlPointGrid(nifti_image *controlPointGridIn) {
        controlPointGrid = controlPointGridIn;
    }
    virtual void SetTransformationGradient(nifti_image *transformationGradientIn) {
        transformationGradient = transformationGradientIn;
    }
    virtual void SetVoxelBasedMeasureGradient(nifti_image *voxelBasedMeasureGradientIn) {
        voxelBasedMeasureGradient = voxelBasedMeasureGradientIn;
    }
    virtual void SetWarpedGradient(nifti_image *warpedGradientIn) {
        warpedGradient = warpedGradientIn;
    }

    // Auxiliary methods
    virtual void ZeroTransformationGradient();
    virtual void ZeroVoxelBasedMeasureGradient();

protected:
    nifti_image *controlPointGrid;
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