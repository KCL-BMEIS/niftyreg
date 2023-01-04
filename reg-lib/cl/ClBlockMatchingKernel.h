#pragma once

#include "BlockMatchingKernel.h"
#include "ClAladinContent.h"

class ClBlockMatchingKernel: public BlockMatchingKernel {
public:
    ClBlockMatchingKernel(Content *conIn);
    ~ClBlockMatchingKernel();
    void Calculate();

private:
    ClContextSingleton *sContext;
    nifti_image *reference;
    _reg_blockMatchingParam *params;
    cl_kernel kernel;
    cl_context clContext;
    cl_program program;
    cl_command_queue commandQueue;
    cl_mem clTotalBlock;
    cl_mem clReferenceImageArray;
    cl_mem clWarpedImageArray;
    cl_mem clReferencePosition;
    cl_mem clWarpedPosition;
    cl_mem clMask;
    cl_mem clReferenceMat;
};
