#ifndef CONTEXT_H_
#define CONTEXT_H_



#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "nifti1_io.h"
#include "_reg_blockMatching.h"

#define CPUX_PLATFORM 0
#define CUDA_PLATFORM 1
#define OCLX_PLATFORM 2



class  Context {
public:

	Context();
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int percentageOfBlocks, const unsigned int  InlierLts, int BlockStepSize/*, bool symmetric*/);
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte);

	virtual ~Context();



	void shout();

	//Platform* platform;

	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateWarpedImage(nifti_image** warpedIn, nifti_image* refIn, nifti_image* floatIn, size_t bytes);
	void ClearWarpedImage();
	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateDeformationField(nifti_image** defFieldIn, nifti_image* refIn, size_t bytes);
	void ClearDeformationField();
	void initVars(const unsigned int platformFlagIn);

	nifti_image* CurrentReference;
	nifti_image* CurrentFloating;
	int* CurrentReferenceMask;
	bool bm;
	bool symmetric;

	mat44* transformationMatrix;
	_reg_blockMatchingParam* blockMatchingParams;


	//getters
	virtual nifti_image* getCurrentDeformationField(){
		return this->CurrentDeformationField;
	}


	nifti_image* getCurrentReference(){
		return  this->CurrentReference;
	}
	nifti_image* getCurrentFloating(){
		return  this->CurrentFloating;
	}
	virtual nifti_image* getCurrentWarped(){
		return CurrentWarped;
	}
	int* getCurrentReferenceMask(){
		return  this->CurrentReferenceMask;
	}
	mat44* getTransformationMatrix(){
		return this->transformationMatrix;
	}
	virtual _reg_blockMatchingParam* getBlockMatchingParams(){
		//std::cout << "serve bm params from cpu" << std::endl;
		return blockMatchingParams;
	}

	//setters
	virtual void setTransformationMatrix(mat44* transformationMatrixIn){
		transformationMatrix = transformationMatrixIn;
	}

	virtual void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn){
		//std::cout << "from context setCurrentDeformationField" << std::endl;
		//nifti_image_free(CurrentDeformationFieldIn);
		CurrentDeformationField = CurrentDeformationFieldIn;
	}

	virtual void setCurrentWarped(nifti_image* CurrentWarpedImageIn){
		//std::cout << "from context" << std::endl;
		//nifti_image_free(CurrentWarped);
		CurrentWarped = CurrentWarpedImageIn;
	}

	//private:
	nifti_image* CurrentDeformationField;
	nifti_image* CurrentWarped;

	nifti_image* CurrentBackwardsDeformationField;
	nifti_image* CurrentBackwardsWarped;





};


#endif /*CONTEXT_H_*/
