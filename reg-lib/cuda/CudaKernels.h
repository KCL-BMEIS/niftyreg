#pragma once
#include "Kernels.h"
#include "CudaContext.h"

//Kernel functions for affine deformation field 
class CudaAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CudaAffineDeformationFieldKernel(Context* conIn, std::string nameIn) : AffineDeformationFieldKernel(nameIn){
		con = ((CudaContext*)conIn);
		this->deformationFieldImage = con->CurrentDeformationField;
		this->affineTransformation = con->transformationMatrix;

		mask_d = con->getMask_d();
		deformationFieldArray_d = con->getDeformationFieldArray_d();

	}

	void execute(bool compose = false);

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;

	float *deformationFieldArray_d;
	int* mask_d;
	CudaContext* con;

};
//Kernel functions for block matching
class CudaBlockMatchingKernel : public BlockMatchingKernel {
public:


	CudaBlockMatchingKernel(Context* conIn, std::string name) : BlockMatchingKernel(name) {
		target = conIn->CurrentReference;
		result = conIn->CurrentWarped;
		params = conIn->blockMatchingParams;
		mask = conIn->CurrentReferenceMask;

		targetImageArray_d = ((CudaContext*)conIn)->getReferenceImageArray_d();
		resultImageArray_d = ((CudaContext*)conIn)->getWarpedImageArray_d();
		targetPosition_d = ((CudaContext*)conIn)->getTargetPosition_d();
		resultPosition_d = ((CudaContext*)conIn)->getResultPosition_d();

		activeBlock_d = ((CudaContext*)conIn)->getActiveBlock_d();
		mask_d = ((CudaContext*)conIn)->getMask_d();
		con = ((CudaContext*)conIn);
	}

	void execute();
	nifti_image* target;
	nifti_image* result;
	_reg_blockMatchingParam* params;
	int* mask;

	CudaContext* con;


	float *targetImageArray_d, *resultImageArray_d, *targetPosition_d, *resultPosition_d;
	int *activeBlock_d, *mask_d;

};
//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel : public ConvolutionKernel {
public:

	CudaConvolutionKernel(std::string name) : ConvolutionKernel(name) {
	}

	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);

};

//kernel functions for numerical optimisation
class CudaOptimiseKernel : public OptimiseKernel {
public:



	CudaOptimiseKernel(Context* conIn, std::string name) : OptimiseKernel(name) {
		con = static_cast<CudaContext*>(conIn);
		transformationMatrix = con->transformationMatrix;
		blockMatchingParams = con->blockMatchingParams;

	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CudaContext *con;
	void execute(bool affine);
};

//kernel functions for image resampling with three interpolation variations
class CudaResampleImageKernel : public ResampleImageKernel {
public:
	CudaResampleImageKernel(Context* conIn, std::string name) : ResampleImageKernel(name) {

		floatingImage = conIn->CurrentFloating;
		warpedImage = conIn->CurrentWarped;
		deformationField = conIn->CurrentDeformationField;
		mask = conIn->CurrentReferenceMask;

		con = static_cast<CudaContext*>(conIn);

		floatingImageArray_d = con->getFloatingImageArray_d();
		warpedImageArray_d = con->getWarpedImageArray_d();
		deformationFieldImageArray_d = con->getDeformationFieldArray_d();
		mask_d = con->getMask_d();


	}
	nifti_image *floatingImage;
	nifti_image *warpedImage;
	nifti_image *deformationField;
	int *mask;

	float* floatingImageArray_d;
	float* warpedImageArray_d;
	float* deformationFieldImageArray_d;
	int* mask_d;

	CudaContext *con;

	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

