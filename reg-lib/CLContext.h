#pragma once
#include "Context.h"
class CLContext : public Context
{

public:
	CLContext();
	CLContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int InlierLts, const unsigned int  CurrentReferenceMask) :Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte, InlierLts, CurrentReferenceMask){}
	CLContext(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :Context(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte){}
	~CLContext();


	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateWarpedImage(size_t bytes);
	void ClearWarpedImage();
	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateDeformationField(size_t bytes);
	void ClearDeformationField();
	void initVars(const unsigned int platformFlagIn);


	//getters
	nifti_image* getCurrentDeformationField(){
		return this->CurrentDeformationField;
	}
	//nifti_image* getCurrentWarped(){
	//	return  this->CurrentWarped;
	//}

	nifti_image* getCurrentReference(){
		return  this->CurrentReference;
	}
	nifti_image* getCurrentFloating(){
		return  this->CurrentFloating;
	}
	nifti_image* getCurrentWarped(){
		return CurrentWarped;
	}
	int* getCurrentReferenceMask(){
		return  this->CurrentReferenceMask;
	}
	mat44* getTransformationMatrix(){
		return this->transformationMatrix;
	}
	_reg_blockMatchingParam* getBlockMatchingParams(){
		return blockMatchingParams;
	}

	//setters
	void setTransformationMatrix(mat44* transformationMatrixIn){
		transformationMatrix = transformationMatrixIn;
	}

	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn){
		CurrentDeformationField = CurrentDeformationFieldIn;
	}

	void setCurrentWarped(nifti_image* CurrentWarpedImageIn){
		CurrentWarped = CurrentWarpedImageIn;
	}

private:
	nifti_image* CurrentDeformationField;
	nifti_image* CurrentWarped;
};