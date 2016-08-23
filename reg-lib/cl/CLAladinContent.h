#ifndef CLCONTENT_H_
#define CLCONTENT_H_

#include "AladinContent.h"
#include "ClGlobalContent.h"
#include "CLContextSingletton.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class ClAladinContent: public AladinContent, public ClGlobalContent {

public:
	//constructors
    ClAladinContent();
	virtual ~ClAladinContent();

    void InitBlockMatchingParams();
    virtual void ClearBlockMatchingParams();

	//opencl getters
	cl_mem getReferencePositionClmem();
	cl_mem getWarpedPositionClmem();
	cl_mem getTotalBlockClmem();

	//setters
    virtual void setTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setTransformationMatrix(mat44 transformationMatrixIn);
    virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp);

    //cpu getters with data downloaded from device
    _reg_blockMatchingParam* getBlockMatchingParams();

protected:
    cl_mem referencePositionClmem;
    cl_mem warpedPositionClmem;
    cl_mem totalBlockClmem;

private:
    //void uploadContext();
    //void allocateClPtrs();
    void freeClPtrs();


};

#endif //CLCONTENT_H_
