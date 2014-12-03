#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "_reg_blockMatching.h"

class Context {
public:

	Context();
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int percentageOfBlocks, const unsigned int InlierLts, int BlockStepSize);
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte);
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte, const unsigned int percentageOfBlocks, const unsigned int InlierLts, int BlockStepSize);
	Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn,mat44* transMat, size_t byte);

	virtual ~Context();

	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateWarpedImage();
	void ClearWarpedImage();
	/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	void AllocateDeformationField(size_t bytes);
	void ClearDeformationField();
	virtual void initVars();

	unsigned int floatingVoxels, referenceVoxels;

	//getters
	virtual nifti_image* getCurrentDeformationField() {
		return this->CurrentDeformationField;
	}

	nifti_image* getCurrentReference() {
		return this->CurrentReference;
	}
	nifti_image* getCurrentFloating() {
		return this->CurrentFloating;
	}
	virtual nifti_image* getCurrentWarped(int datatype = 0) {
		return CurrentWarped;
	}
	int* getCurrentReferenceMask() {
		return this->CurrentReferenceMask;
	}
	mat44* getTransformationMatrix() {
		return this->transformationMatrix;
	}
	int getFloatingDatatype(){
		return this->floatingDatatype;
	}
	virtual _reg_blockMatchingParam* getBlockMatchingParams() {
		//std::cout << "serve bm params from cpu" << std::endl;
		return blockMatchingParams;
	}

	//setters
	virtual void setTransformationMatrix(mat44* transformationMatrixIn) {
		transformationMatrix = transformationMatrixIn;
	}

	virtual void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
		CurrentDeformationField = CurrentDeformationFieldIn;
	}

	virtual void setCurrentWarped(nifti_image* CurrentWarpedImageIn) {
		CurrentWarped = CurrentWarpedImageIn;
	}
	virtual void setCurrentReferenceMask(int* maskIn, size_t nvox) {
	}

	//private:
	nifti_image* CurrentDeformationField;
	nifti_image* CurrentWarped;

	nifti_image* CurrentReference;
	nifti_image* CurrentFloating;
	int* CurrentReferenceMask;

	mat44* transformationMatrix;
	mat44 refMatrix_xyz;
	mat44 floMatrix_ijk;
	_reg_blockMatchingParam* blockMatchingParams;

	 int stepSizeBlock, floatingDatatype;
	unsigned int currentPercentageOfBlockToUse;
	unsigned int inlierLts;
	size_t bytes;

};

#endif /*CONTEXT_H_*/
