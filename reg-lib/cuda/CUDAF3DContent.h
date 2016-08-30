#ifndef CUDAF3DCONTENT_H
#define CUDAF3DCONTENT_H

#include "F3DContent.h"
#include "CUDAGlobalContent.h"

class CudaF3DContent : public F3DContent, public CudaGlobalContent
{
public:
    CudaF3DContent(int refTime, int floTime);
    virtual ~CudaF3DContent();

    //cuda getters
    float* getControlPointGrid_d();
    //cpu getters with data downloaded from device
    virtual nifti_image* getCurrentControlPointGrid(int datatype);
    virtual void setCurrentControlPointGrid(nifti_image *cpgIn);
    virtual void ClearControlPointGrid();

protected:
    float* controlPointGrid_d;
};

#endif // CUDAF3DCONTENT_H
