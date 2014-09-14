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



class  Context {
public:

	Context();
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int InlierLts, const unsigned int  CurrentReferenceMask );
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte);

	~Context();



	/**
	* Set the current time of the simulation (in picoseconds).
	*/
	void setTime(double time);


	void shout();
	

	
	Platform* platform;

	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateWarpedImage(size_t bytes);
	void ClearWarpedImage();
	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateDeformationField(size_t bytes);
	void ClearDeformationField();
	virtual void initVars(const unsigned int platformFlagIn);


	//getters
	nifti_image* getCurrentDeformationField(){
		return this->CurrentDeformationField;
	}
	nifti_image* getCurrenbtWarped(){
		return  this->CurrentWarped;
	}

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
	nifti_image* CurrentReference;
	nifti_image* CurrentFloating;
	int* CurrentReferenceMask;

	mat44* transformationMatrix;
	_reg_blockMatchingParam* blockMatchingParams;
	



};


#endif /*CONTEXT_H_*/
