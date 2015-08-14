#ifndef CLRESAMPLEIMAGEKERNEL_H
#define CLRESAMPLEIMAGEKERNEL_H

#include "ResampleImageKernel.h"
#include "CLContent.h"

class CLResampleImageKernel : public ResampleImageKernel
{
    public:

       CLResampleImageKernel(Content * conIn, std::string name);
       ~CLResampleImageKernel();

       void calculate(int interp, float paddingValue, bool * dti_timepoint = NULL, mat33 * jacMat = NULL);
    private:

       nifti_image *floatingImage;
       nifti_image *warpedImage;
       int *mask;
       CLContextSingletton *sContext;
       ClContent *con;
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

#endif // CLRESAMPLEIMAGEKERNEL_H
