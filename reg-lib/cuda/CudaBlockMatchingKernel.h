#pragma once

#include "../BlockMatchingKernel.h"
#include "CudaAladinContent.h"

//Kernel functions for block matching
class CudaBlockMatchingKernel : public BlockMatchingKernel {
public:

    CudaBlockMatchingKernel(AladinContent *conIn, std::string name);
    void Calculate();
private:
    nifti_image *reference;
    _reg_blockMatchingParam* params;

    //CudaContextSingleton *cudaSContext;
    //CUContext *cudaContext;

    CudaAladinContent *con;

    float *referenceImageArray_d, *warpedImageArray_d, *referencePosition_d;
    float *warpedPosition_d, *referenceMat_d;
    int   *totalBlock_d, *mask_d;

};
