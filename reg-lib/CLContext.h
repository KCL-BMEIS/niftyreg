#pragma once
#include "Context.h"

class ClContext : public Context
{

public:
	ClContext(){
		//std::cout << "Cl context constructor called(empty)" << std::endl;

		initVars();
		allocateClPtrs();
		uploadContext();
	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int  inlierLts, int blockStep) :Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte, blockPercentage, inlierLts, blockStep){
		//std::cout << "Cl context constructor called: " <<bm<< std::endl;
		initVars();
		allocateClPtrs();
		uploadContext();


	}
	ClContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte){
		//std::cout << "Cl (small) context constructor called3" << std::endl;

		initVars();
		allocateClPtrs();
		uploadContext();
	}
	~ClContext();


	float* getReferenceImageArray_d(){
		return referenceImageArray_d;
	}
	float* getFloatingImageArray_d(){
		return floatingImageArray_d;
	}
	float* getWarpedImageArray_d(){
		return warpedImageArray_d;
	}

	float* getTargetPosition_d(){
		return targetPosition_d;
	}
	float* getResultPosition_d(){
		return resultPosition_d;
	}
	float* getDeformationFieldArray_d(){
		return deformationFieldArray_d;
	}
	int* getActiveBlock_d(){
		return activeBlock_d;
	}
	int* getMask_d(){
		return mask_d;
	}


	int* getReferenceDims(){
		return referenceDims;
	}
	int* getFloatingDims(){
		return floatingDims;
	}


	void downloadFromClContext();

	_reg_blockMatchingParam* getBlockMatchingParams();
	nifti_image* getCurrentDeformationField();
	nifti_image* getCurrentWarped();



	void setTransformationMatrix(mat44* transformationMatrixIn);
	void setCurrentWarped(nifti_image* warpedImageIn);
	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn);

private:
	void initVars();


	void uploadContext();
	void allocateClPtrs();
	void freeClPtrs();


	unsigned int numBlocks;

	float *referenceImageArray_d;
	float *floatingImageArray_d;
	float *warpedImageArray_d;
	float *deformationFieldArray_d;
	float *targetPosition_d;
	float *resultPosition_d;
	int *activeBlock_d, *mask_d;

	int referenceDims[4];
	int floatingDims[4];

	unsigned int nVoxels;
};
