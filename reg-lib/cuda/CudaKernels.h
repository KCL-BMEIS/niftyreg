#pragma once
#include "Kernels.h"
#include "CudaContent.h"

//Kernel functions for affine deformation field 
class CudaAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
	CudaAffineDeformationFieldKernel(Content* conIn, std::string nameIn);
	void calculate(bool compose = false);
	void compare(bool compose);
private:
	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;

	float *deformationFieldArray_d, *transformationMatrix_d;
	int* mask_d;
	CudaContent* con;

};
//Kernel functions for block matching
class CudaBlockMatchingKernel: public BlockMatchingKernel {
public:

	CudaBlockMatchingKernel(Content* conIn, std::string name);
	void calculate(int range);
	void compare();
private:
	nifti_image* target;
	_reg_blockMatchingParam* params;

	CudaContent* con;

	float *targetImageArray_d, *resultImageArray_d, *targetPosition_d, *resultPosition_d, *targetMat_d;
	int *activeBlock_d, *mask_d;

};
//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel: public ConvolutionKernel {
public:

	CudaConvolutionKernel(std::string name) :
			ConvolutionKernel(name) {
	}

	void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);

};

//kernel functions for numerical optimisation
class CudaOptimiseKernel: public OptimiseKernel {
public:

	CudaOptimiseKernel(Content* conIn, std::string name);
	void calculate(bool affine, bool ils);
private:
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CudaContent *con;

	float* transformationMatrix_d;
	float* AR_d;
	float* U_d;
	float* Sigma_d;
	float* VT_d;
	float* lengths_d;
	float* targetPos_d;
	float* resultPos_d;
	float* newResultPos_d;

};

/*
 * kernel functions for image resampling with three interpolation variations
 * */
class CudaResampleImageKernel: public ResampleImageKernel {
public:
	CudaResampleImageKernel(Content* conIn, std::string name);
	void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
private:
	nifti_image *floatingImage;
	nifti_image *warpedImage;


	//cuda ptrs
	float* floatingImageArray_d;
	float* floIJKMat_d;
	float* warpedImageArray_d;
	float* deformationFieldImageArray_d;
	int* mask_d;

	CudaContent *con;


};

