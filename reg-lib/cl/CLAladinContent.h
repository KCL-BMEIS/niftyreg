#ifndef CLCONTENT_H_
#define CLCONTENT_H_

#include "AladinContent.h"
#include "CLContextSingletton.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class ClAladinContent: public AladinContent {

public:

	//constructors
    ClAladinContent();
	virtual ~ClAladinContent();

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();

    void InitBlockMatchingParams();
    virtual void ClearBlockMatchingParams();

    bool isCurrentComputationDoubleCapable();

	//opencl getters
	cl_mem getReferenceImageArrayClmem();
	cl_mem getFloatingImageArrayClmem();
	cl_mem getWarpedImageClmem();
	cl_mem getReferencePositionClmem();
	cl_mem getWarpedPositionClmem();
	cl_mem getDeformationFieldArrayClmem();
	cl_mem getTotalBlockClmem();
	cl_mem getMaskClmem();
	cl_mem getRefMatClmem();
	cl_mem getFloMatClmem();
	int *getReferenceDims();
	int *getFloatingDims();

	//cpu getters with data downloaded from device
    nifti_image* getCurrentWarped(int datatype);
    nifti_image* getCurrentDeformationField();
    _reg_blockMatchingParam* getBlockMatchingParams();

	//setters
    virtual void setCurrentReference(nifti_image* currentRefIn);
    virtual void setCurrentReferenceMask(int *maskIn, size_t size);
    virtual void setCurrentFloating(nifti_image* currentFloIn);
    virtual void setCurrentWarped(nifti_image *warpedImageIn);
    virtual void setCurrentDeformationField(nifti_image *currentDeformationFieldIn);

    virtual void setTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setTransformationMatrix(mat44 transformationMatrixIn);
    virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp);


private:
    //void uploadContext();
    //void allocateClPtrs();
	void freeClPtrs();

	CLContextSingletton *sContext;
	cl_context clContext;
	cl_int errNum;
	cl_command_queue commandQueue;

	cl_mem referenceImageClmem;
	cl_mem floatingImageClmem;
	cl_mem warpedImageClmem;
	cl_mem deformationFieldClmem;
	cl_mem referencePositionClmem;
	cl_mem warpedPositionClmem;
	cl_mem totalBlockClmem;
	cl_mem maskClmem;
	cl_mem refMatClmem;
	cl_mem floMatClmem;

	int referenceDims[4];
	int floatingDims[4];

	unsigned int nVoxels;

	void downloadImage(nifti_image *image,
							 cl_mem memoryObject,
							 int datatype);
	template<class T>
	void fillImageData(nifti_image *image,
							 cl_mem memoryObject,
							 int type);
	template<class T>
	T fillWarpedImageData(float intensity,
								 int datatype);

};

#endif //CLCONTENT_H_
