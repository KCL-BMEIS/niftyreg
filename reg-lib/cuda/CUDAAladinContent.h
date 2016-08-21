#pragma once

#include "AladinContent.h"
#include "CUDAContextSingletton.h"

#include "_reg_tools.h"

class CudaAladinContent: public AladinContent {

public:
	CudaAladinContent();
    virtual ~CudaAladinContent();

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();

    void InitBlockMatchingParams();
    virtual void ClearBlockMatchingParams();

	bool isCurrentComputationDoubleCapable();

	//device getters
	float* getReferenceImageArray_d();
	float* getFloatingImageArray_d();
	float* getWarpedImageArray_d();
	float* getTransformationMatrix_d();
	float* getReferencePosition_d();
	float* getWarpedPosition_d();
	float* getDeformationFieldArray_d();
	float* getReferenceMat_d();
	float* getFloIJKMat_d();

	//	float* getAR_d(); // Removed until CUDA SVD is added back
	//	float* getU_d(); // Removed until CUDA SVD is added back
	//	float* getVT_d(); // Removed until CUDA SVD is added back
	//	float* getSigma_d(); // Removed until CUDA SVD is added back
	//	float* getLengths_d(); // Removed until CUDA SVD is added back
	//	float* getNewWarpedPos_d(); // Removed until CUDA SVD is added back

	int *getTotalBlock_d();
	int *getMask_d();

	int *getReferenceDims();
	int *getFloatingDims();

	//cpu getters and setters
    nifti_image *getCurrentWarped(int typ);
    nifti_image *getCurrentDeformationField();
	_reg_blockMatchingParam* getBlockMatchingParams();

    //setters
    virtual void setCurrentReference(nifti_image* currentRefIn);
    virtual void setCurrentReferenceMask(int *maskIn, size_t size);
    virtual void setCurrentFloating(nifti_image* currentFloIn);
    virtual void setCurrentWarped(nifti_image *warpedImageIn);
    virtual void setCurrentDeformationField(nifti_image *currentDeformationFieldIn);

    virtual void setTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setTransformationMatrix(mat44 transformationMatrixIn);
    virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp);

private:
	void initVars();

	//void uploadAladinContent();
    //void allocateCuPtrs();
	void freeCuPtrs();

	CUDAContextSingletton* cudaSContext;
	CUcontext cudaContext;

	float *referenceImageArray_d;
	float *floatingImageArray_d;
	float *warpedImageArray_d;
	float *deformationFieldArray_d;
	float *referencePosition_d;
	float *warpedPosition_d;
	int   *totalBlock_d, *mask_d;

	float* transformationMatrix_d;
	float* referenceMat_d;
	float* floIJKMat_d;

	//svd
	//	float* AR_d;//A and then pseudoinverse  // Removed until CUDA SVD is added back
	//	float* U_d; // Removed until CUDA SVD is added back
	//	float* VT_d; // Removed until CUDA SVD is added back
	//	float* Sigma_d; // Removed until CUDA SVD is added back
	//	float* lengths_d; // Removed until CUDA SVD is added back
	//	float* newWarpedPos_d; // Removed until CUDA SVD is added back

	int referenceDims[4];
	int floatingDims[4];

	void downloadImage(nifti_image *image, float* memoryObject, int datatype);
	template<class T>
	void fillImageData(nifti_image *image, float* memoryObject, int type);

	template<class FloatingTYPE>
	FloatingTYPE fillWarpedImageData(float intensity, int datatype);
};
