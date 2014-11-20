#ifndef KERNELS_H_
#define KERNELS_H_

#include <iosfwd>
#include <string>
#include <vector>

#include "Kernel.h"
#include "nifti1_io.h"


class AffineDeformationFieldKernel : public Kernel {
public:
	static std::string Name() {
		return "AffineDeformationFieldKernel";
	}
	AffineDeformationFieldKernel( std::string name) : Kernel(name) {
	}

	virtual void execute(bool compose = false) = 0;
};

class BlockMatchingKernel : public Kernel {
public:
	static std::string Name() {
		return "blockMatchingKernel";
	}
	BlockMatchingKernel(std::string name) : Kernel(name) {

	}

	virtual void execute() = 0;
};


class ConvolutionKernel : public Kernel {
public:
	static std::string Name() {
		return "ConvolutionKernel";
	}
	ConvolutionKernel(std::string name) : Kernel(name) {
	}

	virtual void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
};

class OptimiseKernel : public Kernel{
public:
	static std::string Name() {
		return "OptimiseKernel";
	}
	OptimiseKernel(std::string name) : Kernel(name) {
	}
	virtual void execute(bool affine) = 0;
};
class ResampleImageKernel : public Kernel {
public:
	static std::string Name() {
		return "ResampleImageKernel";
	}
	ResampleImageKernel( std::string name) : Kernel(name) {

	}
	virtual ~ResampleImageKernel(){std::cout<<"virtual ResampleImageKernel"<<std::endl;}

	virtual void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
};
#endif /*KERNELS_H_*/
