#ifndef CUDABLOCKMATCHINGKERNEL_H
#define CUDABLOCKMATCHINGKERNEL_H

#include "../BlockMatchingKernel.h"
#include "CUDAAladinContent.h"

//Kernel functions for block matching
class CUDABlockMatchingKernel : public BlockMatchingKernel {
public:

    CUDABlockMatchingKernel(AladinContent *conIn, std::string name);
    void calculate();
private:
    nifti_image *reference;
    _reg_blockMatchingParam* params;

    //CUDAContextSingletton *cudaSContext;
    //CUContext *cudaContext;

    CudaAladinContent *con;

    float *referenceImageArray_d, *warpedImageArray_d, *referencePosition_d;
    float *warpedPosition_d, *referenceMat_d;
    int   *totalBlock_d, *mask_d;

};

#endif // CUDABLOCKMATCHINGKERNEL_H
