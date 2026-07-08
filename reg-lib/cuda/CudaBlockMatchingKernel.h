#pragma once

#include "../BlockMatchingKernel.h"
#include "CudaAladinContent.h"

//Kernel functions for block matching
class CudaBlockMatchingKernel: public BlockMatchingKernel {
public:
    explicit CudaBlockMatchingKernel(Content *conIn);
    virtual void Calculate() override;

private:
    nifti_image *reference;
    _reg_blockMatchingParam *params;

    float *referenceCuda, *warpedCuda, *referencePositionCuda;
    float *warpedPositionCuda, *referenceMatCuda;
    int *totalBlockCuda, *maskCuda;
};
