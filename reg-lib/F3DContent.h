#ifndef F3DCONTENT_H
#define F3DCONTENT_H

#include "GlobalContent.h"
#include "_reg_localTrans.h"

class F3DContent : public virtual GlobalContent
{
public:
    F3DContent(int platformCodeIn, int refTime, int floTime);
    virtual ~F3DContent();
    //
    void setInputControlPointGrid(nifti_image* cpg);
    nifti_image* getInputControlPointGrid();
    //Final spacing
    void setSpacing(unsigned int i, float s);
    float* getSpacing();
#ifdef BUILD_DEV
    bool getLinearSpline();
    void setLinearSpline(bool ls);
#endif
    //
    virtual void setCurrentControlPointGrid(nifti_image* cpg);
    virtual nifti_image* getCurrentControlPointGrid();
    virtual void AllocateControlPointGrid(float* gridSpacing);
    virtual void AllocateControlPointGrid();
    virtual void ClearControlPointGrid();

protected:
    bool linearSpline;
    float* spacing;
    nifti_image* inputControlPointGrid;
    nifti_image* currentControlPointGrid;
};

#endif // F3DCONTENT_H
