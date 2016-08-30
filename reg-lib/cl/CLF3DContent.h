#ifndef CLF3DCONTENT_H
#define CLF3DCONTENT_H

#include "F3DContent.h"
#include "CLGlobalContent.h"

class ClF3DContent : public F3DContent, public ClGlobalContent
{
public:
    ClF3DContent(int refTime, int floTime);
    virtual ~ClF3DContent();

    //opencl getters
    cl_mem getControlPointGridClmem();
    //cpu getters with data downloaded from device
    virtual nifti_image* getCurrentControlPointGrid(int datatype);
    virtual void setCurrentControlPointGrid(nifti_image *cpgIn);
    virtual void ClearControlPointGrid();

protected:
    cl_mem controlPointGridClmem;
};

#endif // CLF3DCONTENT_H
