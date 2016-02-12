#ifndef ALADINCONTENT_H_
#define ALADINCONTENT_H_

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
					  const unsigned int InlierLts,
					  int BlockStepSize);
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
					  const unsigned int InlierLts,
					  int BlockStepSize);
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
	virtual void initVars();

	unsigned int floatingVoxels, referenceVoxels;

	//getters
	virtual nifti_image *getCurrentDeformationField()
	{
		return this->CurrentDeformationField;
	}
	nifti_image *getCurrentReference()
	{
		return this->CurrentReference;
	}
	nifti_image *getCurrentFloating()
	{
		return this->CurrentFloating;
	}
	virtual nifti_image *getCurrentWarped(int = 0)
	{
		return this->CurrentWarped;
	}
	int *getCurrentReferenceMask()
	{
		return this->CurrentReferenceMask;
	}
	mat44 *getTransformationMatrix()
	{
		return this->transformationMatrix;
	}
	virtual _reg_blockMatchingParam* getBlockMatchingParams() {
		return this->blockMatchingParams;
	}
	//setters
	virtual void setTransformationMatrix(mat44 *transformationMatrixIn)
	{
		this->transformationMatrix = transformationMatrixIn;
	}
	virtual void setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
	{
		this->CurrentDeformationField = CurrentDeformationFieldIn;
	}
	virtual void setCurrentWarped(nifti_image *CurrentWarpedImageIn)
	{
		this->CurrentWarped = CurrentWarpedImageIn;
	}

	virtual void setCurrentReferenceMask(int *, size_t) {}
	void setCaptureRange(const int captureRangeIn);
	//
	virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp) {
		blockMatchingParams = bmp;
	}

	virtual bool isCurrentComputationDoubleCapable();

protected:
	nifti_image *CurrentReference;
	nifti_image *CurrentFloating;
	int *CurrentReferenceMask;

	nifti_image *CurrentDeformationField;
	nifti_image *CurrentWarped;

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

#endif //ALADINCONTENT_H_
