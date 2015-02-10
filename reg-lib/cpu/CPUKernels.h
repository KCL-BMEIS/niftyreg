#ifndef CPUKERNELS_H
#define CPUKERNELS_H

#include "Kernels.h"
#include "Content.h"


/* *************************************************************** */
//Kernel functions for affine deformation field
class CPUAffineDeformationFieldKernel : public AffineDeformationFieldKernel
{
public:
	CPUAffineDeformationFieldKernel(Content *con, std::string nameIn) : AffineDeformationFieldKernel( nameIn) {
		this->deformationFieldImage = con->getCurrentDeformationField();
		this->affineTransformation = con->getTransformationMatrix();
		this->mask = con->getCurrentReferenceMask();
	}


	void calculate(bool compose = false);

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int *mask;
};
/* *************************************************************** */
//Kernel functions for block matching
class CPUBlockMatchingKernel : public BlockMatchingKernel
{
public:

	CPUBlockMatchingKernel(Content *con, std::string name) : BlockMatchingKernel( name) {
		reference = con->getCurrentReference();
		warped = con->getCurrentWarped();
		params = con->getBlockMatchingParams();
		mask = con->getCurrentReferenceMask();
	}

	void calculate();
	nifti_image *reference;
	nifti_image *warped;
	_reg_blockMatchingParam* params;
	int *mask;

};
/* *************************************************************** */
//a kernel function for convolution (gaussian smoothing?)
class CPUConvolutionKernel : public ConvolutionKernel
{
public:

	CPUConvolutionKernel(std::string name) : ConvolutionKernel(name) {
	}

	void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
};
/* *************************************************************** */
//kernel functions for numerical optimisation
class CPUOptimiseKernel : public OptimiseKernel
{
public:

	CPUOptimiseKernel(Content *con, std::string name) : OptimiseKernel( name) {
		transformationMatrix = con->getTransformationMatrix();
		blockMatchingParams = con->getBlockMatchingParams();
	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;

	void calculate(bool affine, bool ils, bool svd=0);
};
/* *************************************************************** */
//kernel functions for image resampling with three interpolation variations
class CPUResampleImageKernel : public ResampleImageKernel
{
public:
	CPUResampleImageKernel(Content *con, std::string name) : ResampleImageKernel( name) {
		floatingImage = con->getCurrentFloating();
		warpedImage = con->getCurrentWarped();
		deformationField = con->getCurrentDeformationField();
		mask = con->getCurrentReferenceMask();
	}
	nifti_image *floatingImage;
	nifti_image *warpedImage;
	nifti_image *deformationField;
	int *mask;

	void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};
/* *************************************************************** */

#endif
