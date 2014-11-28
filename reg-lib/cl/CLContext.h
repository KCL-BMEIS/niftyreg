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

		initVars();
		allocateClPtrs();
	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte, blockPercentage, inlierLts, blockStep) {
		//std::cout << "Cl context constructor called: " <<bm<< std::endl;

//		std::cout<<"CL Context Constructor Init"<<std::endl;
		initVars();
		allocateClPtrs();
//		std::cout<<"CL Context Constructor End"<<std::endl;
	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte) {
//		std::cout<<"CL Context Constructor Init"<<std::endl;
		initVars();
//		std::cout<<"CL Context Init Vars"<<std::endl;
		allocateClPtrs();
//		std::cout<<"CL Context Constructor End"<<std::endl;
	}

	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn,mat44* transMat, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn,transMat, byte, blockPercentage, inlierLts, blockStep) {
		//std::cout << "Cl context constructor called: " <<bm<< std::endl;

		//		std::cout<<"CL Context Constructor Init"<<std::endl;
		initVars();
		allocateClPtrs();
		//		std::cout<<"CL Context Constructor End"<<std::endl;
	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn,mat44* transMat, size_t byte) :
			Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn,transMat, byte) {
		//		std::cout<<"CL Context Constructor Init"<<std::endl;
		initVars();
		//		std::cout<<"CL Context Init Vars"<<std::endl;
		allocateClPtrs();
		//		std::cout<<"CL Context Constructor End"<<std::endl;
	}
	~ClContext();

	CLContextSingletton *sContext;
	cl_context clContext;
	cl_int errNum;
	cl_command_queue commandQueue;

	cl_mem getReferenceImageArrayClmem() {
		return referenceImageClmem;
	}
	cl_mem getFloatingImageArrayClmem() {
		return floatingImageClmem;
	}
	cl_mem getWarpedImageClmem() {
		return warpedImageClmem;
	}

	cl_mem getTargetPositionClmem() {
		return targetPositionClmem;
	}
	cl_mem getResultPositionClmem() {
		return resultPositionClmem;
	}
	cl_mem getDeformationFieldArrayClmem() {
		return deformationFieldClmem;
	}
	cl_mem getActiveBlockClmem() {
		return activeBlockClmem;
	}
	cl_mem getMaskClmem() {
		return maskClmem;
	}
	cl_mem getRefMatClmem() {
		return refMatClmem;
	}
	cl_mem getFloMatClmem() {
		return floMatClmem;
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
	nifti_image* getCurrentWarped(int typ);

	void setTransformationMatrix(mat44* transformationMatrixIn);
	void setCurrentWarped(nifti_image* warpedImageIn);
	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn);
	void setCurrentReferenceMask(int* maskIn, size_t size);
//	void checkErrNum(cl_int errNum, std::string message);

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

	float* warpedImageBuffer;

};

#endif //CLCONTEXT_H_
