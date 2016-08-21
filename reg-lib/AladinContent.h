#ifndef ALADINCONTENT_H_
#define ALADINCONTENT_H_

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "_reg_blockMatching.h"
#include "GlobalContent.h"

class AladinContent : public GlobalContent
{
public:

    AladinContent(int platformCodeIn);
    virtual ~AladinContent();

    virtual void InitBlockMatchingParams();

    //getters
    //setters
    void setCaptureRange(const int captureRangeIn);
    void setPercentageOfBlock(unsigned pob);
    unsigned getPercentageOfBlock();
    void setInlierLts(unsigned ilts);
    unsigned getInlierLts();
    void setBlockStepSize(int bss);
    int getBlockStepSize();
    virtual void setTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setTransformationMatrix(mat44 transformationMatrixIn);
    mat44* getTransformationMatrix();
    virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp);
    virtual _reg_blockMatchingParam* getBlockMatchingParams();

    virtual void ClearBlockMatchingParams();

    virtual bool isCurrentComputationDoubleCapable();

protected:

    mat44 *transformationMatrix;
	_reg_blockMatchingParam* blockMatchingParams;

    //unsigned int floatingVoxels, referenceVoxels;
	//int floatingDatatype;
    size_t bytes;
};

#endif //ALADINCONTENT_H_
