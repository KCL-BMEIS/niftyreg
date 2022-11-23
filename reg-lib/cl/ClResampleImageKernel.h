#pragma once

#include "ResampleImageKernel.h"
#include "CLAladinContent.h"

class ClResampleImageKernel : public ResampleImageKernel
{
    public:

       ClResampleImageKernel(AladinContent * conIn, std::string name);
       ~ClResampleImageKernel();

       void Calculate(int interp, float paddingValue, bool * dti_timepoint = nullptr, mat33 * jacMat = nullptr);
    private:

       nifti_image *floatingImage;
       nifti_image *warpedImage;
       int *mask;
       ClContextSingleton *sContext;
       ClAladinContent *con;
       cl_command_queue commandQueue;
       cl_kernel kernel;
       cl_context clContext;
       cl_program program;
       cl_mem clCurrentFloating;
       cl_mem clCurrentDeformationField;
       cl_mem clCurrentWarped;
       cl_mem clMask;
       cl_mem floMat;
};
