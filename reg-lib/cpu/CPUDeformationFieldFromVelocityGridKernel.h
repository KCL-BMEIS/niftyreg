#ifndef CPUDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
#define CPUDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H

#include "DeformationFieldFromVelocityGridKernel.h"
#include "F3DContent.h"
#include "_reg_localTrans.h"

class CPUDeformationFieldFromVelocityGridKernel : public DeformationFieldFromVelocityGridKernel
{
public:
    CPUDeformationFieldFromVelocityGridKernel(GlobalContent *con, std::string nameIn);
    void calculate(bool updateStepNumber = true);

private:
    F3DContent* con;
    nifti_image* deformationFieldImage;
    nifti_image* controlPointImage;
};

#endif // CPUDEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
