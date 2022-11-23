#pragma once

#include "AffineDeformationFieldKernel.h"
#include "CLAladinContent.h"

class ClAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
    public:
       ClAffineDeformationFieldKernel(AladinContent * conIn, std::string nameIn);
       ~ClAffineDeformationFieldKernel();

       void Calculate(bool compose = false);
    private:
       mat44 *affineTransformation, *ReferenceMatrix;
       nifti_image *deformationFieldImage;
       ClAladinContent *con;
       cl_command_queue commandQueue;
       cl_kernel kernel;
       cl_context clContext;
       cl_program program;
       cl_mem clDeformationField, clMask;
       ClContextSingleton *sContext;
};
