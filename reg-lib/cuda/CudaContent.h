#pragma once
#include "Content.h"
#include "_reg_tools.h"

class CudaContent: public Content {

public:
	CudaContent();
	CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep);
	CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte);
	CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep);
	CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte);
	~CudaContent();

	//device getters
	float* getReferenceImageArray_d();
	float* getFloatingImageArray_d();
	float* getWarpedImageArray_d();
	float* getTransformationMatrix_d();
	float* getTargetPosition_d();
	float* getResultPosition_d();
	float* getDeformationFieldArray_d();
	float* getTargetMat_d();
	float* getFloIJKMat_d();
	int* getActiveBlock_d();
	int* getMask_d();

	int* getReferenceDims();
	int* getFloatingDims();



	//cpu getters and setters
	_reg_blockMatchingParam* getBlockMatchingParams();
	nifti_image* getCurrentDeformationField();
	nifti_image* getCurrentWarped(int typ);

	void setTransformationMatrix(mat44* transformationMatrixIn);
	void setCurrentWarped(nifti_image* warpedImageIn);
	void setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn);
	void setCurrentReferenceMask(int* maskIn, size_t size);

private:
	void initVars();

	void uploadContent();
	void allocateCuPtrs();
	void freeCuPtrs();

	unsigned int numBlocks;

	float *referenceImageArray_d;
	float *floatingImageArray_d;
	float *warpedImageArray_d;
	float *deformationFieldArray_d;
	float *targetPosition_d;
	float *resultPosition_d;
	int *activeBlock_d, *mask_d;

	float* transformationMatrix_d;
	float* targetMat_d;
	float* floIJKMat_d;

	int referenceDims[4];
	int floatingDims[4];

	void downloadImage(nifti_image* image, float* memoryObject, bool flag, int datatype, std::string message);
	template<class T>
	void fillImageData(nifti_image* image, float* memoryObject, bool warped, int type, std::string message);

	template<class FloatingTYPE>
	FloatingTYPE fillWarpedImageData(float intensity, int datatype);

	unsigned long nVoxels;

};
