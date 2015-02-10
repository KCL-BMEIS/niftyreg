#ifndef CONTENT_H_
#define CONTENT_H_

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "_reg_blockMatching.h"

class Content {
public:

	Content();
	Content(nifti_image *CurrentReferenceIn,
			  nifti_image *CurrentFloatingIn,
			  int *CurrentReferenceMaskIn,
			  size_t byte,
			  const unsigned int percentageOfBlocks,
			  const unsigned int InlierLts,
			  int BlockStepSize);
	Content(nifti_image *CurrentReferenceIn,
			  nifti_image *CurrentFloatingIn,
			  int *CurrentReferenceMaskIn,
			  size_t byte);
	Content(nifti_image *CurrentReferenceIn,
			  nifti_image *CurrentFloatingIn,
			  int *CurrentReferenceMaskIn,
			  mat44 *transMat,
			  size_t byte,
			  const unsigned int percentageOfBlocks,
			  const unsigned int InlierLts,
			  int BlockStepSize);
	Content(nifti_image *CurrentReferenceIn,
			  nifti_image *CurrentFloatingIn,
			  int *CurrentReferenceMaskIn,
			  mat44 *transMat,
			  size_t byte);

	virtual ~Content();

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
		return CurrentWarped;
	}
	int *getCurrentReferenceMask()
	{
		return this->CurrentReferenceMask;
	}
	mat44 *getTransformationMatrix()
	{
		return this->transformationMatrix;
	}
	int getFloatingDatatype()
	{
		return this->floatingDatatype;
	}
	virtual _reg_blockMatchingParam* getBlockMatchingParams()
	{
		return blockMatchingParams;
	}
	//setters
	virtual void setTransformationMatrix(mat44 *transformationMatrixIn)
	{
		transformationMatrix = transformationMatrixIn;
	}
	virtual void setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
	{
		CurrentDeformationField = CurrentDeformationFieldIn;
	}
	virtual void setCurrentWarped(nifti_image *CurrentWarpedImageIn)
	{
		CurrentWarped = CurrentWarpedImageIn;
	}
	virtual void setCurrentReferenceMask(int *, size_t) {}
	void setCaptureRange(const int captureRangeIn);

protected:
	nifti_image *CurrentDeformationField;
	nifti_image *CurrentWarped;

	nifti_image *CurrentReference;
	nifti_image *CurrentFloating;
	int *CurrentReferenceMask;

	mat44 *transformationMatrix;
	mat44 refMatrix_xyz;
	mat44 floMatrix_ijk;
	_reg_blockMatchingParam* blockMatchingParams;

	int floatingDatatype;
	size_t bytes;
	unsigned int currentPercentageOfBlockToUse;
	unsigned int inlierLts;
	int stepSizeBlock;
};

#endif /*CONTENT_H_*/
