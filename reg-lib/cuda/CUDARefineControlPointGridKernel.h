#ifndef CUDAREFINECONTROLPOINTGRIDKERNEL_H
#define CUDAREFINECONTROLPOINTGRIDKERNEL_H

#include "RefineControlPointGridKernel.h"
#include "CUDAF3DContent.h"
#include "_reg_localTrans.h"

class CUDARefineControlPointGridKernel : public RefineControlPointGridKernel
{
    public:
        CUDARefineControlPointGridKernel(GlobalContent *con, std::string nameIn);
        void calculate();

    private:
        F3DContent* con;
        nifti_image* controlPointImage;
        nifti_image* referenceImage;
};

#endif // CUDAREFINECONTROLPOINTGRIDKERNEL_H
