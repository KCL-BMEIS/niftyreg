#pragma once
#include "kernels.h"
#include "context.h"


class CudaAffineDeformationFieldKernel;
class CudaBlockMatchingKernel;
class CudaConvolutionKernel;
class CudaResampleImageKernel;


//Kernel functions for affine deformation field 
class CudaAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CudaAffineDeformationFieldKernel(Context* con, std::string nameIn, const Platform& platformIn) : AffineDeformationFieldKernel(nameIn, platformIn){
		this->deformationFieldImage = con->getCurrentDeformationField();
		this->affineTransformation = con->getTransformationMatrix();
		this->mask = con->getCurrentReferenceMask();
	}


	void execute(bool compose = false);

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int* mask;

	/*template<class FieldTYPE> void runKernel3D(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
	template <class FieldTYPE> void runKernel2D(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool compose, int *mask);*/


};
//Kernel functions for block matching
class CudaBlockMatchingKernel : public BlockMatchingKernel {
public:

	CudaBlockMatchingKernel(Context* con, std::string name, const Platform& platform) : BlockMatchingKernel(name, platform) {
		target = con->getCurrentReference();
		result = con->getCurrentWarped();
		params = con->getBlockMatchingParams();
		mask = con->getCurrentReferenceMask();
	}

	void execute();
	nifti_image* target;
	nifti_image* result;
	_reg_blockMatchingParam* params;
	int* mask;

};
//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel : public ConvolutionKernel {
public:

	CudaConvolutionKernel(std::string name, const Platform& platform) : ConvolutionKernel(name, platform) {
	}

	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);

};

//kernel functions for numerical optimisation
class CudaOptimiseKernel : public OptimiseKernel {
public:

	CudaOptimiseKernel(Context* con, std::string name, const Platform& platform) : OptimiseKernel(name, platform) {
		transformationMatrix = con->getTransformationMatrix();
		blockMatchingParams = con->getBlockMatchingParams();
	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	void execute(bool affine);
};

//kernel functions for image resampling with three interpolation variations
class CudaResampleImageKernel : public ResampleImageKernel {
public:
	CudaResampleImageKernel(Context* con, std::string name, const Platform& platform) : ResampleImageKernel(name, platform) {
		floatingImage = con->getCurrentFloating();
		warpedImage = con->getCurrentWarped();
		deformationField = con->getCurrentDeformationField();
		mask = con->getCurrentReferenceMask();
	}
	nifti_image *floatingImage;
	nifti_image *warpedImage;
	nifti_image *deformationField;
	int *mask;


	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

