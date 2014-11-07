#pragma once

#include "kernels.h"
#include "CLContextSingletton.h"
#include "Context.h"



class CLOptimiseKernel;
class CLAffineDeformationFieldKernel;
class CLBlockMatchingKernel;
class CLConvolutionKernel;
class CLResampleImageKernel;


//Kernel functions for affine deformation field 
class CLAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CLAffineDeformationFieldKernel(Context* con, std::string nameIn) : AffineDeformationFieldKernel(nameIn) {
		sContext = &CLContextSingletton::Instance();

		this->deformationFieldImage = con->getCurrentDeformationField();
		this->affineTransformation = con->getTransformationMatrix();
		this->mask = con->getCurrentReferenceMask();
	}


	void execute(bool compose = false);

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int* mask;
	CLContextSingletton* sContext;

};
//Kernel functions for block matching
class CLBlockMatchingKernel : public BlockMatchingKernel {
public:

	CLBlockMatchingKernel(Context* con, std::string name) : BlockMatchingKernel(name) {
		sContext = &CLContextSingletton::Instance();
		target = con->getCurrentReference();
		result = con->getCurrentWarped();
		params = con->getBlockMatchingParams();
		mask = con->getCurrentReferenceMask();
	}
	CLContextSingletton* sContext;
	void execute();

	nifti_image* target;
	nifti_image* result;
	_reg_blockMatchingParam* params;
	int* mask;


};
//a kernel function for convolution (gaussian smoothing?)
class CLConvolutionKernel : public ConvolutionKernel {
public:

	CLConvolutionKernel(std::string name) : ConvolutionKernel(name) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL){}
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) {}

};

//kernel functions for numerical optimisation
class CLOptimiseKernel : public OptimiseKernel {
public:

	CLOptimiseKernel(Context* con, std::string name) : OptimiseKernel( name) {
		sContext = &CLContextSingletton::Instance();
		transformationMatrix = con->getTransformationMatrix();
		blockMatchingParams = con->getBlockMatchingParams();
	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CLContextSingletton* sContext;
	void execute(bool affine) {}
};

//kernel functions for image resampling with three interpolation variations
class CLResampleImageKernel : public ResampleImageKernel {
public:
	CLResampleImageKernel(Context* con, std::string name) : ResampleImageKernel( name) {
		sContext = &CLContextSingletton::Instance();
		floatingImage = con->getCurrentFloating();
		warpedImage = con->getCurrentWarped();
		deformationField = con->getCurrentDeformationField();
		mask = con->getCurrentReferenceMask();
	}
	nifti_image *floatingImage;
	nifti_image *warpedImage;
	nifti_image *deformationField;
	int *mask;
	CLContextSingletton* sContext;

	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

