#ifndef CUDADEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
#define CUDADEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H

#include "DeformationFieldFromVelocityGridKernel.h"
#include "CUDAF3DContent.h"
#include "_reg_localTrans.h"

class CUDADeformationFieldFromVelocityGridKernel : public DeformationFieldFromVelocityGridKernel
{
    public:
        CUDADeformationFieldFromVelocityGridKernel(GlobalContent *con, std::string nameIn);
        void calculate(bool updateStepNumber = true);

    private:
        F3DContent* con;
        nifti_image* deformationFieldImage;
        nifti_image* controlPointImage;
};

#endif // CUDADEFORMATIONFIELDFROMVELOCITYGRIDKERNEL_H
