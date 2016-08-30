#ifndef CLREFINECONTROLPOINTGRIDKERNEL_H
#define CLREFINECONTROLPOINTGRIDKERNEL_H

#include "RefineControlPointGridKernel.h"
#include "CLF3DContent.h"
#include "_reg_localTrans.h"

class ClRefineControlPointGridKernel : public RefineControlPointGridKernel
{
public:
    ClRefineControlPointGridKernel(GlobalContent *con, std::string nameIn);
    void calculate();

private:
    F3DContent* con;
    nifti_image* controlPointImage;
    nifti_image* referenceImage;
};

#endif // CLREFINECONTROLPOINTGRIDKERNEL_H
