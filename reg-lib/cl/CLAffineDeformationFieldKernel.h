#ifndef CLAFFINEDEFORMATIONFIELDKERNEL_H
#define CLAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"
#include "CLAladinContent.h"

class CLAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
    public:
       CLAffineDeformationFieldKernel(AladinContent * conIn, std::string nameIn);
       ~CLAffineDeformationFieldKernel();

       void calculate(bool compose = false);
    private:
       mat44 *affineTransformation, *ReferenceMatrix;
       nifti_image *deformationFieldImage;
       ClAladinContent *con;
       cl_command_queue commandQueue;
       cl_kernel kernel;
       cl_context clContext;
       cl_program program;
       cl_mem clDeformationField, clMask;
       CLContextSingletton *sContext;
};

#endif // CLAFFINEDEFORMATIONFIELDKERNEL_H
