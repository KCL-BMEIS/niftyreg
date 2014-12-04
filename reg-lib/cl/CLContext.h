#ifndef CLCONTEXT_H_
#define CLCONTEXT_H_

#include "Context.h"
#include "CLContextSingletton.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class ClContext: public Context {

public:

	//constructors
	ClContext();
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep);
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte);
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep);
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte);
	~ClContext();

	//opencl getters
	cl_mem getReferenceImageArrayClmem();
	cl_mem getFloatingImageArrayClmem();
	cl_mem getWarpedImageClmem();
	cl_mem getTargetPositionClmem();
	cl_mem getResultPositionClmem();
	cl_mem getDeformationFieldArrayClmem();
	cl_mem getActiveBlockClmem();
	cl_mem getMaskClmem();
	cl_mem getRefMatClmem();
	cl_mem getFloMatClmem();
	int* getReferenceDims();
	int* getFloatingDims();

	//cpu getters with data downloaded from device
	_reg_blockMatchingParam* getBlockMatchingParams();
	nifti_image* getCurrentDeformationField();
	nifti_image* getCurrentWarped(int typ);

	//setters
	void setTransformationMatrix(mat44* transformationMatrixIn);
	void setCurrentWarped(nifti_image* warpedImageIn);
	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn);
	void setCurrentReferenceMask(int* maskIn, size_t size);


private:
	void initVars();

	void uploadContext();
	void allocateClPtrs();
	void freeClPtrs();

	unsigned int numBlocks;
	CLContextSingletton *sContext;
	cl_context clContext;
	cl_int errNum;
	cl_command_queue commandQueue;

	cl_mem referenceImageClmem;
	cl_mem floatingImageClmem;
	cl_mem warpedImageClmem;
	cl_mem deformationFieldClmem;
	cl_mem targetPositionClmem;
	cl_mem resultPositionClmem;
	cl_mem activeBlockClmem, maskClmem;
	cl_mem refMatClmem, floMatClmem;

	int referenceDims[4];
	int floatingDims[4];

	unsigned int nVoxels;

	void downloadImage(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int datatype, std::string message);
	template<class T>
	void fillImageData(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
	template<class FloatingTYPE>
	FloatingTYPE fillWarpedImageData(float intensity, int datatype);

};

#endif //CLCONTEXT_H_
