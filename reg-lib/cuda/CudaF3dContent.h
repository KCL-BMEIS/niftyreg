#pragma once

#include "F3dContent.h"
#include "CudaContent.h"

class CudaF3dContent: public F3dContent, public CudaContent {
public:
    CudaF3dContent() = delete;
    CudaF3dContent(nifti_image *referenceIn,
                   nifti_image *floatingIn,
                   nifti_image *controlPointGridIn,
                   nifti_image *localWeightSimIn = nullptr,
                   int *referenceMaskIn = nullptr,
                   mat44 *transformationMatrixIn = nullptr,
                   size_t bytesIn = sizeof(float));
    virtual ~CudaF3dContent();

    // Getters
    virtual nifti_image* GetControlPointGrid() override;
    virtual nifti_image* GetTransformationGradient() override;
    virtual nifti_image* GetVoxelBasedMeasureGradient() override;
    virtual nifti_image* GetWarpedGradient() override;
    virtual float4* GetControlPointGridCuda() { return controlPointGridCuda; }
    virtual float4* GetTransformationGradientCuda() { return transformationGradientCuda; }
    virtual float4* GetVoxelBasedMeasureGradientCuda() { return voxelBasedMeasureGradientCuda; }
    virtual float4* GetWarpedGradientCuda() { return warpedGradientCuda; }

    // Methods for transferring data from nifti to device
    virtual void UpdateControlPointGrid() override;
    virtual void UpdateTransformationGradient() override;
    virtual void UpdateVoxelBasedMeasureGradient() override;
    virtual void UpdateWarpedGradient() override;

    // Auxiliary methods
    virtual void ZeroTransformationGradient() override;
    virtual void ZeroVoxelBasedMeasureGradient() override;

protected:
    float4 *controlPointGridCuda = nullptr;
    float4 *transformationGradientCuda = nullptr;
    float4 *voxelBasedMeasureGradientCuda = nullptr;
    float4 *warpedGradientCuda = nullptr;

private:
    void AllocateControlPointGrid();
    void DeallocateControlPointGrid();
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
};
