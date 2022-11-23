#pragma once

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "_reg_blockMatching.h"

class AladinContent {
public:
	AladinContent();
	AladinContent(nifti_image *CurrentReferenceIn,
				  nifti_image *CurrentFloatingIn,
				  int *CurrentReferenceMaskIn,
				  size_t byte,
				  const unsigned int percentageOfBlocks,
				  const unsigned int inlierLts,
				  int blockStepSize);
	AladinContent(nifti_image *CurrentReferenceIn,
				  nifti_image *CurrentFloatingIn,
				  int *CurrentReferenceMaskIn,
				  size_t byte);
	AladinContent(nifti_image *CurrentReferenceIn,
				  nifti_image *CurrentFloatingIn,
				  int *CurrentReferenceMaskIn,
				  mat44 *transMat,
				  size_t byte,
				  const unsigned int percentageOfBlocks,
				  const unsigned int inlierLts,
				  int blockStepSize);
	AladinContent(nifti_image *CurrentReferenceIn,
				  nifti_image *CurrentFloatingIn,
				  int *CurrentReferenceMaskIn,
				  mat44 *transMat,
				  size_t byte);

	virtual ~AladinContent();

	/* *************************************************************** */
	void AllocateWarpedImage();
	void ClearWarpedImage();
	/* *************************************************************** */
	void AllocateDeformationField(size_t bytes);
	void ClearDeformationField();
	virtual void InitVars();

	unsigned int floatingVoxels, referenceVoxels;

	//getters
	virtual nifti_image* GetCurrentDeformationField() {
		return this->currentDeformationField;
	}
	nifti_image* GetCurrentReference() {
		return this->currentReference;
	}
	nifti_image* GetCurrentFloating() {
		return this->currentFloating;
	}
	virtual nifti_image* GetCurrentWarped(int = 0) {
		return this->currentWarped;
	}
	int* GetCurrentReferenceMask() {
		return this->currentReferenceMask;
	}
	mat44* GetTransformationMatrix() {
		return this->transformationMatrix;
	}
	virtual _reg_blockMatchingParam* GetBlockMatchingParams() {
		return this->blockMatchingParams;
	}
	//setters
	virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) {
		this->transformationMatrix = transformationMatrixIn;
	}
	virtual void SetCurrentDeformationField(nifti_image *CurrentDeformationFieldIn) {
		this->currentDeformationField = CurrentDeformationFieldIn;
	}
	virtual void SetCurrentWarped(nifti_image *CurrentWarpedImageIn) {
		this->currentWarped = CurrentWarpedImageIn;
	}

	virtual void SetCurrentReferenceMask(int *, size_t) {}
	void SetCaptureRange(const int captureRangeIn);
	//
	virtual void SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
		blockMatchingParams = bmp;
	}

	virtual bool IsCurrentComputationDoubleCapable();

protected:
	nifti_image *currentReference;
	nifti_image *currentFloating;
	int *currentReferenceMask;

	nifti_image *currentDeformationField;
	nifti_image *currentWarped;

	mat44 *transformationMatrix;
	mat44 refMatrix_xyz;
	mat44 floMatrix_ijk;
	_reg_blockMatchingParam* blockMatchingParams;

	//int floatingDatatype;
	size_t bytes;
	unsigned int currentPercentageOfBlockToUse;
	unsigned int inlierLts;
	int stepSizeBlock;
};
