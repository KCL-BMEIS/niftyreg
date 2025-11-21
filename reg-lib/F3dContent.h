#pragma once

#include "DefContent.h"

class F3dContent: public virtual DefContent {
public:
    F3dContent() = delete;
    F3dContent(NiftiImage& referenceIn,
               NiftiImage& floatingIn,
               NiftiImage& controlPointGridIn,
               NiftiImage *localWeightSimIn = nullptr,
               int *referenceMaskIn = nullptr,
               mat44 *transformationMatrixIn = nullptr,
               size_t bytesIn = sizeof(float));

    // Getters
    virtual NiftiImage& GetControlPointGrid() { return controlPointGrid; }
    virtual NiftiImage& GetTransformationGradient() { return transformationGradient; }

    // Methods for transferring data from nifti to device
    virtual void UpdateControlPointGrid() {}
    virtual void UpdateTransformationGradient() {}

    // Auxiliary methods
    virtual void ZeroTransformationGradient();

protected:
    NiftiImage controlPointGrid;
    NiftiImage transformationGradient;
};