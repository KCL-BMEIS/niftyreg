#pragma once

#include "AffineDeformationFieldKernel.h"
#include "ClAladinContent.h"

class ClAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
    ClAffineDeformationFieldKernel(Content *conIn);
    ~ClAffineDeformationFieldKernel();
    void Calculate(bool compose = false);

private:
    mat44 *affineTransformation, *referenceMatrix;
    nifti_image *deformationFieldImage;
    cl_command_queue commandQueue;
    cl_kernel kernel;
    cl_context clContext;
    cl_program program;
    cl_mem clDeformationField, clMask;
    ClContextSingleton *sContext;
};
