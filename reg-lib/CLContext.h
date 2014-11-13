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
	ClContext() {
		//std::cout << "Cl context constructor called(empty)" << std::endl;
		sContext = &CLContextSingletton::Instance();
		initVars();
		allocateClPtrs();
	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte, blockPercentage, inlierLts, blockStep) {
		//std::cout << "Cl context constructor called: " <<bm<< std::endl;
		sContext = &CLContextSingletton::Instance();

		initVars();
		allocateClPtrs();

	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte) {
//		std::cout << "Cl (small) context constructor called3" << std::endl;
		sContext = &CLContextSingletton::Instance();
		initVars();
		allocateClPtrs();

	}
	~ClContext();

	CLContextSingletton *sContext;
	cl_context clContext;
	cl_int errNum;
	cl_command_queue commandQueue;

	cl_mem getReferenceImageArray_d() {
		return referenceImageClmem;
	}
	cl_mem getFloatingImageArray_d() {
		return floatingImageClmem;
	}
	cl_mem getWarpedImageArray_d() {
		return warpedImageClmem;
	}

	cl_mem getTargetPosition_d() {
		return targetPosition_d;
	}
	cl_mem getResultPosition_d() {
		return resultPosition_d;
	}
	cl_mem getDeformationFieldArray_d() {
		return deformationFieldClmem;
	}
	cl_mem getActiveBlock_d() {
		return activeBlock_d;
	}
	cl_mem getMask_d() {
		return mask_d;
	}

	int* getReferenceDims() {
		return referenceDims;
	}
	int* getFloatingDims() {
		return floatingDims;
	}

	void downloadFromClContext();

	_reg_blockMatchingParam* getBlockMatchingParams();
	nifti_image* getCurrentDeformationField();
	nifti_image* getCurrentWarped();

	void setTransformationMatrix(mat44* transformationMatrixIn);
	void setCurrentWarped(nifti_image* warpedImageIn);
	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn);
	void checkErrNum(cl_int errNum, std::string message);

private:
	void initVars();

	void uploadContext();
	void allocateClPtrs();
	void freeClPtrs();

	unsigned int numBlocks;

	cl_mem referenceImageClmem;
	cl_mem floatingImageClmem;
	cl_mem warpedImageClmem;
	cl_mem deformationFieldClmem;
	cl_mem targetPosition_d;
	cl_mem resultPosition_d;
	cl_mem activeBlock_d, mask_d;

	float* referenceBuffer;
	float* floatingBuffer;

	float* warpedBuffer;
	float* deformationFieldBuffer;

	int referenceDims[4];
	int floatingDims[4];

	unsigned int nVoxels;

	template<class T>
	void fillBuffer(float** buffer, T* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, std::string message);

	void uploadImage(float** buffer, nifti_image* image, cl_mem* memoryObject, cl_mem_flags flag, bool keep, std::string message);
	void downloadImage(float* buffer, nifti_image* image, cl_mem memoryObject, cl_mem_flags flag,  std::string message);


	void fillBuffers();

	template<class T>
	void fillImageData(float* buffer, T* array, size_t size, cl_mem memoryObject, cl_mem_flags flag,  std::string message);
};

#endif //CLCONTEXT_H_
