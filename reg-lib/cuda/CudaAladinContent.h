#pragma once

#include "AladinContent.h"
#include "CudaContent.h"
#include "CudaContext.hpp"
#include "_reg_tools.h"

class CudaAladinContent: public virtual AladinContent, public virtual CudaContent {
public:
    CudaAladinContent() = delete;
    CudaAladinContent(NiftiImage& referenceIn,
                      NiftiImage& floatingIn,
                      int *referenceMaskIn = nullptr,
                      mat44 *transformationMatrixIn = nullptr,
                      size_t bytesIn = sizeof(float),
                      const unsigned percentageOfBlocks = 0,
                      const unsigned inlierLts = 0,
                      int blockStepSize = 0);
    virtual ~CudaAladinContent() = default;

    virtual bool IsCurrentComputationDoubleCapable() override;

    // Getters
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() override;
    virtual float* GetReferencePositionCuda() { return referencePositionCuda.get(); }
    virtual float* GetWarpedPositionCuda() { return warpedPositionCuda.get(); }
    virtual float* GetReferenceMatCuda() { return referenceMatCuda.get(); }
    virtual int* GetTotalBlockCuda() { return totalBlockCuda.get(); }
    // Dense per-voxel mask used by the affine and block-matching kernels;
    // CudaContent holds the compacted active-voxel list used by the resampler
    virtual int* GetMaskCuda() { return maskCuda.get(); }

protected:
    Cuda::UniquePtr<float> referencePositionCuda;
    Cuda::UniquePtr<float> warpedPositionCuda;
    Cuda::UniquePtr<float> referenceMatCuda;
    Cuda::UniquePtr<int> totalBlockCuda;
    Cuda::UniquePtr<int> maskCuda;  // Dense per-voxel mask (compacted list lives in CudaContent)

private:
    void AllocateMask();
    void AllocateReferenceMat();
    void AllocateBlockMatchingParams();

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam *bmp) override;
};
