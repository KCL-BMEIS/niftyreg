#pragma once
#include "kernels.h"
#include "Context.h"
#include "CudaContext.h"


class CudaAffineDeformationFieldKernel;
class CudaBlockMatchingKernel;
class CudaConvolutionKernel;
class CudaResampleImageKernel;


//Kernel functions for affine deformation field 
class CudaAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CudaAffineDeformationFieldKernel(Context* conIn, std::string nameIn) : AffineDeformationFieldKernel(nameIn){
		this->deformationFieldImage = conIn->getCurrentDeformationField();
		this->affineTransformation = conIn->getTransformationMatrix();
		this->mask = conIn->getCurrentReferenceMask();

		mask_d = ((CudaContext*)conIn)->getMask_d();
		deformationFieldArray_d = ((CudaContext*)conIn)->getDeformationFieldArray_d();
		con = ((CudaContext*)conIn);
	}


	void execute(bool compose = false);

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int* mask;

	float *deformationFieldArray_d;
	int* mask_d;
	CudaContext* con;
	/*template<class FieldTYPE> void runKernel3D(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
	template <class FieldTYPE> void runKernel2D(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool compose, int *mask);*/


};
//Kernel functions for block matching
class CudaBlockMatchingKernel : public BlockMatchingKernel {
public:

	CudaBlockMatchingKernel(Context* conIn, std::string name) : BlockMatchingKernel(name) {
		target = conIn->getCurrentReference();
		result = conIn->getCurrentWarped();
		params = conIn->getBlockMatchingParams();
		mask = conIn->getCurrentReferenceMask();

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
		transformationMatrix = conIn->getTransformationMatrix();
		blockMatchingParams = conIn->getBlockMatchingParams();
		con = static_cast<CudaContext*>(conIn);
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
		floatingImage = conIn->getCurrentFloating();
		warpedImage = conIn->getCurrentWarped();
		deformationField = conIn->getCurrentDeformationField();
		mask = conIn->getCurrentReferenceMask();

		floatingImageArray_d = ((CudaContext*)conIn)->getFloatingImageArray_d();
		warpedImageArray_d = ((CudaContext*)conIn)->getWarpedImageArray_d();
		deformationFieldImageArray_d = ((CudaContext*)conIn)->getDeformationFieldArray_d();
		mask_d = ((CudaContext*)conIn)->getMask_d();

		con = static_cast<CudaContext*>(conIn);
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

