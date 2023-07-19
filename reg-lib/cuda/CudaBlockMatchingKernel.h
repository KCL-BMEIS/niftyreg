#pragma once

#include "../BlockMatchingKernel.h"
#include "CudaAladinContent.h"

//Kernel functions for block matching
class CudaBlockMatchingKernel: public BlockMatchingKernel {
public:
    explicit CudaBlockMatchingKernel(Content *conIn);
    void Calculate();

private:
    nifti_image *reference;
    _reg_blockMatchingParam *params;

    float *referenceImageArray_d, *warpedImageArray_d, *referencePosition_d;
    float *warpedPosition_d, *referenceMat_d;
    int *totalBlock_d, *mask_d;
};
