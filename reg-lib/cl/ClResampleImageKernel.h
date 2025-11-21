#pragma once

#include "ResampleImageKernel.h"
#include "ClAladinContent.h"

class ClResampleImageKernel: public ResampleImageKernel {
public:
    ClResampleImageKernel(Content *conIn);
    ~ClResampleImageKernel();
    void Calculate(int interp, float paddingValue, bool *dtiTimePoint = nullptr, mat33 *jacMat = nullptr);

private:
    nifti_image *floatingImage;
    nifti_image *warpedImage;
    int *mask;
    ClContextSingleton *sContext;
    cl_command_queue commandQueue;
    cl_kernel kernel;
    cl_context clContext;
    cl_program program;
    cl_mem clFloating;
    cl_mem clDeformationField;
    cl_mem clWarped;
    cl_mem clMask;
    cl_mem floMat;
};
