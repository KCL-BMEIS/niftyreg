#pragma once

#include "DefContent.h"
#include "CudaContent.h"

class CudaDefContent: public virtual DefContent, public virtual CudaContent {
public:
    CudaDefContent() = delete;
    CudaDefContent(nifti_image *referenceIn,
                   nifti_image *floatingIn,
                   nifti_image *localWeightSimIn = nullptr,
                   int *referenceMaskIn = nullptr,
                   mat44 *transformationMatrixIn = nullptr,
                   size_t bytesIn = sizeof(float));
    virtual ~CudaDefContent();

    // Getters
    virtual nifti_image* GetLocalWeightSim() override;
    virtual nifti_image* GetVoxelBasedMeasureGradient() override;
    virtual nifti_image* GetWarpedGradient() override;
    virtual float* GetLocalWeightSimCuda() { return localWeightSimCuda; }
    virtual float4* GetVoxelBasedMeasureGradientCuda() { return voxelBasedMeasureGradientCuda; }
    virtual float4* GetWarpedGradientCuda() { return warpedGradientCuda; }

    // Methods for transferring data from nifti to device
    virtual void UpdateVoxelBasedMeasureGradient() override;
    virtual void UpdateWarpedGradient() override;

    // Auxiliary methods
    virtual void ZeroVoxelBasedMeasureGradient() override;

protected:
    float *localWeightSimCuda = nullptr;
    float4 *voxelBasedMeasureGradientCuda = nullptr;
    float4 *warpedGradientCuda = nullptr;

private:
    void AllocateLocalWeightSim();
    void DeallocateLocalWeightSim();
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
};
