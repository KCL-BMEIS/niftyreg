#ifndef CLSPLINEDEFORMATIONFIELDKERNEL_H
#define CLSPLINEDEFORMATIONFIELDKERNEL_H

#include "SplineDeformationFieldKernel.h"
#include "CLF3DContent.h"
#include "_reg_localTrans.h"

class ClSplineDeformationFieldKernel : public SplineDeformationFieldKernel
{
public:
    ClSplineDeformationFieldKernel(GlobalContent *con, std::string nameIn);
    void calculate(bool compose = false);

private:
    F3DContent* con;
    nifti_image* deformationFieldImage;
    nifti_image* controlPointImage;
    int* mask;
};

#endif // CLSPLINEDEFORMATIONFIELDKERNEL_H
