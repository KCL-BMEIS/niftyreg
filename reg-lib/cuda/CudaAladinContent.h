#pragma once

#include "AladinContent.h"
#include "CudaContent.h"
#include "CudaContext.hpp"
#include "_reg_tools.h"

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

    // Getters
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() override;
    virtual float* GetReferencePositionCuda() { return referencePositionCuda; }
    virtual float* GetWarpedPositionCuda() { return warpedPositionCuda; }
    virtual float* GetReferenceMatCuda() { return referenceMatCuda; }
    virtual int* GetTotalBlockCuda() { return totalBlockCuda; }
    // Dense per-voxel mask used by the affine and block-matching kernels;
    // CudaContent holds the compacted active-voxel list used by the resampler
    virtual int* GetMaskCuda() { return maskCuda; }

private:
    void InitVars();
    void AllocateCuPtrs();
    void FreeCuPtrs();

    float *referencePositionCuda;
    float *warpedPositionCuda;
    float *referenceMatCuda;
    int *totalBlockCuda;
    int *maskCuda;  // Dense per-voxel mask (compacted list lives in CudaContent)

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam *bmp) override;
};
