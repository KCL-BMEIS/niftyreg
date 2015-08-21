#ifndef CUDABLOCKMATCHINGKERNEL_H
#define CUDABLOCKMATCHINGKERNEL_H

#include "BlockMatchingKernel.h"

//Kernel functions for block matching
class CudaBlockMatchingKernel: public BlockMatchingKernel
{
public:

    CudaBlockMatchingKernel(Content *conIn, std::string name);
    void calculate();
private:
    nifti_image *reference;
    _reg_blockMatchingParam* params;

    CudaContent *con;

    float *referenceImageArray_d, *warpedImageArray_d, *referencePosition_d;
    float *warpedPosition_d, *referenceMat_d;
    int *activeBlock_d, *mask_d;

};

#endif // CUDABLOCKMATCHINGKERNEL_H
