#ifndef CPUSPLINEDEFORMATIONFIELDKERNEL_H
#define CPUSPLINEDEFORMATIONFIELDKERNEL_H

#include "SplineDeformationFieldKernel.h"
#include "F3DContent.h"
#include "_reg_localTrans.h"

class CPUSplineDeformationFieldKernel : public SplineDeformationFieldKernel
{
public:
    CPUSplineDeformationFieldKernel(GlobalContent *con, std::string nameIn);
    void calculate(bool compose = false);

private:
    F3DContent* con;
    nifti_image* deformationFieldImage;
    nifti_image* controlPointImage;
    int* mask;
};

#endif // CPUSPLINEDEFORMATIONFIELDKERNEL_H
