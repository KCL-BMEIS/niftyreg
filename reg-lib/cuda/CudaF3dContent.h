#pragma once

#include "F3dContent.h"
#include "CudaDefContent.h"

class CudaF3dContent: public F3dContent, public CudaDefContent {
public:
    CudaF3dContent() = delete;
    CudaF3dContent(NiftiImage& referenceIn,
                   NiftiImage& floatingIn,
                   NiftiImage& controlPointGridIn,
                   NiftiImage *localWeightSimIn = nullptr,
                   int *referenceMaskIn = nullptr,
                   mat44 *transformationMatrixIn = nullptr,
                   size_t bytesIn = sizeof(float));
    virtual ~CudaF3dContent();

    // Getters
    virtual NiftiImage& GetControlPointGrid() override;
    virtual NiftiImage& GetTransformationGradient() override;
    virtual float4* GetControlPointGridCuda() { return controlPointGridCuda; }
    virtual float4* GetTransformationGradientCuda() { return transformationGradientCuda; }

    // Methods for transferring data from nifti to device
    virtual void UpdateControlPointGrid() override;
    virtual void UpdateTransformationGradient() override;

    // Auxiliary methods
    virtual void ZeroTransformationGradient() override;

protected:
    float4 *controlPointGridCuda = nullptr;
    float4 *transformationGradientCuda = nullptr;

private:
    void AllocateControlPointGrid();
    void DeallocateControlPointGrid();
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();
};
