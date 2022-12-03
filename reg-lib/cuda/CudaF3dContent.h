#pragma once

#include "F3dContent.h"
#include "CudaContent.h"
#include "_reg_blocksize_gpu.h"

class CudaF3dContent: public F3dContent, public CudaContent {
public:
    CudaF3dContent() = delete;
    CudaF3dContent(nifti_image *referenceIn,
                   nifti_image *floatingIn,
                   nifti_image *controlPointGridIn,
                   nifti_image *localWeightSimIn,
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
    virtual float4** GetWarpedGradientCuda() { return warpedGradientCuda; }

    // Setters
    virtual void SetControlPointGrid(nifti_image *controlPointGridIn) override;
    virtual void SetTransformationGradient(nifti_image *transformationGradientIn) override;
    virtual void SetVoxelBasedMeasureGradient(nifti_image *voxelBasedMeasureGradientIn) override;
    virtual void SetWarpedGradient(nifti_image *warpedGradientIn) override;

    // Auxiliary methods
    virtual void ZeroTransformationGradient() override;
    virtual void ZeroVoxelBasedMeasureGradient() override;

protected:
    float4 *controlPointGridCuda = nullptr;
    float4 *transformationGradientCuda = nullptr;
    float4 *voxelBasedMeasureGradientCuda = nullptr;
    float4 *warpedGradientCuda[2] = {nullptr};

private:
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
};