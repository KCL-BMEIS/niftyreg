#ifndef KERNELS_H_
#define KERNELS_H_

#include <iosfwd>
#include <string>
#include <vector>

#include "Kernel.h"
#include "nifti1_io.h"


class AffineDeformationFieldKernel : public Kernel {
public:
	static std::string getName() {
		return "AffineDeformationFieldKernel";
	}
	AffineDeformationFieldKernel( std::string name) : Kernel(name) {
	}
	virtual ~AffineDeformationFieldKernel(){}
	virtual void calculate(bool compose = false) = 0;
};

class BlockMatchingKernel : public Kernel {
public:
	static std::string getName() {
		return "blockMatchingKernel";
	}
	BlockMatchingKernel(std::string name) : Kernel(name) {

	}
	virtual ~BlockMatchingKernel(){}
	virtual void calculate() = 0;
};


class ConvolutionKernel : public Kernel {
public:
	static std::string getName() {
		return "ConvolutionKernel";
	}
	ConvolutionKernel(std::string name) : Kernel(name) {
	}
	virtual ~ConvolutionKernel(){}
	virtual void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
};

class OptimiseKernel : public Kernel{
public:
	static std::string getName() {
		return "OptimiseKernel";
	}
	OptimiseKernel(std::string name) : Kernel(name) {
	}
	virtual ~OptimiseKernel(){}
	virtual void calculate(bool affine, bool ils, bool cusvd) = 0;
};
class ResampleImageKernel : public Kernel {
public:
	static std::string getName() {
		return "ResampleImageKernel";
	}
	ResampleImageKernel( std::string name) : Kernel(name) {

	}
	virtual ~ResampleImageKernel(){}

	virtual void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
};
#endif /*KERNELS_H_*/
