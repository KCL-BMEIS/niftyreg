#ifndef KERNELS_H_
#define KERNELS_H_

#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "KernelImpl.h"
#include "nifti1_io.h"
#include "_reg_blockMatching.h"//temporarily

class AffineDeformationFieldKernel : public KernelImpl {
public:
	static std::string Name() {
		return "AffineDeformationFieldKernel";
	}
	AffineDeformationFieldKernel(nifti_image *deformationFieldImageIn, mat44 *affineTransformationIn, std::string name, const Platform& platform, int *maskIn = NULL) : KernelImpl(name, platform) {
		this->deformationFieldImage = deformationFieldImageIn;
		this->affineTransformation = affineTransformationIn;
		this->mask = maskIn;
	}

	virtual void execute(bool compose = false) = 0;

	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int* mask;
};

class BlockMatchingKernel : public KernelImpl {
public:
	static std::string Name() {
		return "blockMatchingKernel";
	}
	BlockMatchingKernel(nifti_image * targetIn, nifti_image * resultIn, _reg_blockMatchingParam *paramsIn, int *maskIn, std::string name, const Platform& platform) : KernelImpl(name, platform) {
		target = targetIn;
		result = resultIn;
		params = paramsIn;
		mask = maskIn;
	}

	nifti_image* target;
	nifti_image* result;
	_reg_blockMatchingParam* params;
	int* mask;

	virtual void execute() = 0;
};


class ConvolutionKernel : public KernelImpl {
public:
	static std::string Name() {
		return "ConvolutionKernel";
	}
	ConvolutionKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void execute(nifti_image *image,
		float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
};

class OptimiseKernel : public KernelImpl{
public:
	static std::string Name() {
		return "OptimiseKernel";
	}
	OptimiseKernel(_reg_blockMatchingParam *paramsIn, mat44 *transformationMatrixIn, std::string name, const Platform& platform) : KernelImpl(name, platform) {
		transformationMatrix = transformationMatrixIn;
		blockMatchingParams = paramsIn;
	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	virtual void execute(bool affine) = 0;
};
class ResampleImageKernel : public KernelImpl {
public:
	static std::string Name() {
		return "ResampleImageKernel";
	}
	ResampleImageKernel(nifti_image *floatingImageIn, nifti_image *warpedImageIn, nifti_image *deformationFieldIn, int *maskIn, std::string name, const Platform& platform) : KernelImpl(name, platform) {
		//std::cout << "new resample kernel instance\n" << std::endl;
		floatingImage = floatingImageIn;
		warpedImage = warpedImageIn;
		deformationField = deformationFieldIn;
		mask = maskIn;
	}
	nifti_image *floatingImage;
	nifti_image *warpedImage;
	nifti_image *deformationField;
	int *mask;

	virtual void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
};
#endif /*KERNELS_H_*/
