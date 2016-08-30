#ifndef CPUREFINECONTROLPOINTGRID_H
#define CPUREFINECONTROLPOINTGRID_H

#include "RefineControlPointGridKernel.h"
#include "F3DContent.h"
#include "_reg_localTrans.h"

class CPURefineControlPointGridKernel : public RefineControlPointGridKernel
{
public:
    CPURefineControlPointGridKernel(GlobalContent *con, std::string nameIn);
    void calculate();

private:
    F3DContent* con;
    nifti_image* controlPointImage;
    nifti_image* referenceImage;
};

#endif // CPUREFINECONTROLPOINTGRID_H
