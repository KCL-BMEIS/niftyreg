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
	AffineDeformationFieldKernel( std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void execute(bool compose = false) = 0;
};

class BlockMatchingKernel : public KernelImpl {
public:
	static std::string Name() {
		return "blockMatchingKernel";
	}
	BlockMatchingKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {

	}

	virtual void execute() = 0;
};


class ConvolutionKernel : public KernelImpl {
public:
	static std::string Name() {
		return "ConvolutionKernel";
	}
	ConvolutionKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
};

class OptimiseKernel : public KernelImpl{
public:
	static std::string Name() {
		return "OptimiseKernel";
	}
	OptimiseKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}
	virtual void execute(bool affine) = 0;
};
class ResampleImageKernel : public KernelImpl {
public:
	static std::string Name() {
		return "ResampleImageKernel";
	}
	ResampleImageKernel( std::string name, const Platform& platform) : KernelImpl(name, platform) {

	}


	virtual void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
};
#endif /*KERNELS_H_*/
