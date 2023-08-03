#pragma once

#include "DefContent.h"

class F3dContent: public virtual DefContent {
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
    virtual nifti_image* GetTransformationGradient() { return transformationGradient; }

    // Methods for transferring data from nifti to device
    virtual void UpdateControlPointGrid() {}
    virtual void UpdateTransformationGradient() {}

    // Auxiliary methods
    virtual void ZeroTransformationGradient();

protected:
    nifti_image *controlPointGrid = nullptr;
    nifti_image *transformationGradient = nullptr;

private:
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();
};