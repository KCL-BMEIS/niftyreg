#ifndef CUDASPLINEDEFORMATIONFIELDKERNEL_H
#define CUDASPLINEDEFORMATIONFIELDKERNEL_H

#include "SplineDeformationFieldKernel.h"
#include "CUDAF3DContent.h"
#include "_reg_localTrans.h"

class CUDASplineDeformationFieldKernel : public SplineDeformationFieldKernel
{
    public:
        CUDASplineDeformationFieldKernel(GlobalContent *con, std::string nameIn);
        void calculate(bool compose = false);

    private:
        F3DContent* con;
        nifti_image* deformationFieldImage;
        nifti_image* controlPointImage;
        int* mask;
};

#endif // CUDASPLINEDEFORMATIONFIELDKERNEL_H
