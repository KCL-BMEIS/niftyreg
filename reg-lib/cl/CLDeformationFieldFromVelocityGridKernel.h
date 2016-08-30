#ifndef CLDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
#define CLDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H

#include "DeformationFieldFromVelocityGridKernel.h"
#include "CLF3DContent.h"
#include "_reg_localTrans.h"

class ClDeformationFieldFromVelocityGridKernel : public DeformationFieldFromVelocityGridKernel
{
public:
    ClDeformationFieldFromVelocityGridKernel(GlobalContent *con, std::string nameIn);
    void calculate(bool updateStepNumber = true);

private:
    F3DContent* con;
    nifti_image* deformationFieldImage;
    nifti_image* controlPointImage;
};

#endif // CLDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
