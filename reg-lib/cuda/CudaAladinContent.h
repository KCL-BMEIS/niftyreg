#pragma once

#include "AladinContent.h"
#include "CudaContent.h"
#include "CudaContext.hpp"
#include "_reg_tools.h"

// Aladin content on CUDA. It reuses CudaContent's device storage (image buffers, the float4
// deformation field, the compacted reference-mask list, the transformation matrix, and the
// upload/download helpers) and only adds the block-matching-specific device buffers plus a DENSE
// per-voxel mask (the affine and block-matching kernels index the mask randomly by voxel, which the
// compacted list cannot serve).
class CudaAladinContent: public virtual AladinContent, public virtual CudaContent {
public:
    CudaAladinContent(NiftiImage& referenceIn,
                      NiftiImage& floatingIn,
                      int *referenceMaskIn = nullptr,
                      mat44 *transformationMatrixIn = nullptr,
                      size_t bytesIn = sizeof(float),
                      const unsigned percentageOfBlocks = 0,
                      const unsigned inlierLts = 0,
                      int blockStepSize = 0);
    virtual ~CudaAladinContent();

    virtual bool IsCurrentComputationDoubleCapable() override;

    // Device getters. The image/transformation getters forward to the shared CudaContent buffers;
    // the deformation field (GetDeformationFieldCuda, float4) and the compacted mask
    // (GetReferenceMaskCuda) also come from CudaContent. The block-matching buffers and the dense
    // mask are Aladin-specific.
    virtual float* GetReferenceImageArray_d() { return GetReferenceCuda(); }
    virtual float* GetFloatingImageArray_d() { return GetFloatingCuda(); }
    virtual float* GetWarpedImageArray_d() { return GetWarpedCuda(); }
    virtual float* GetTransformationMatrix_d() { return GetTransformationMatrixCuda(); }
    virtual float* GetReferencePosition_d() { return referencePosition_d; }
    virtual float* GetWarpedPosition_d() { return warpedPosition_d; }
    virtual float* GetReferenceMat_d() { return referenceMat_d; }
    virtual int* GetTotalBlock_d() { return totalBlock_d; }
    virtual int* GetMask_d() { return mask_d; } // dense per-voxel mask (block matching / affine)

    // CPU getter with data downloaded from device
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() override;

private:
    void InitVars();
    void AllocateCuPtrs();
    void FreeCuPtrs();

    // Aladin-specific device buffers (the shared image/deformation/mask/transform buffers live in
    // CudaContent).
    float *referencePosition_d;
    float *warpedPosition_d;
    int   *totalBlock_d;
    int   *mask_d;         // dense per-voxel mask (compacted list lives in CudaContent)
    float *referenceMat_d;

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam *bmp) override;
};
