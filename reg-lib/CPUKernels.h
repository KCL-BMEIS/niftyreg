#pragma once
#include "kernels.h"
#include"Context.h"



class CPUAffineDeformationFieldKernel;
class CPUBlockMatchingKernel;
class CPUConvolutionKernel;
class CPUOptimiseKernel;
class CPUResampleImageKernel;


//Kernel functions for affine deformation field 
class CPUAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CPUAffineDeformationFieldKernel(Context* con, std::string nameIn, const Platform& platformIn) : AffineDeformationFieldKernel(con->getCurrentDeformationField(), con->getTransformationMatrix(), nameIn, platformIn, con->getCurrentReferenceMask()) {
	}


	void execute(bool compose = false);
};
//Kernel functions for block matching
class CPUBlockMatchingKernel : public BlockMatchingKernel {
public:

	CPUBlockMatchingKernel(Context* con, std::string name, const Platform& platform) : BlockMatchingKernel(con->getCurrentReference(), con->getCurrentWarped(), con->getBlockMatchingParams(), con->getCurrentReferenceMask(), name, platform) {
	}

	void execute();


};
//a kernel function for convolution (gaussian smoothing?)
class CPUConvolutionKernel : public ConvolutionKernel {
public:

	CPUConvolutionKernel(std::string name, const Platform& platform) : ConvolutionKernel(name, platform) {
	}

	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
private:
	bool *nanImagePtr;
	float *densityPtr;

};

//kernel functions for numerical optimisation
class CPUOptimiseKernel : public OptimiseKernel {
public:

	CPUOptimiseKernel(Context* con, std::string name, const Platform& platform) : OptimiseKernel(con->getBlockMatchingParams(), con->getTransformationMatrix(), name, platform) {
	}

	void execute(bool affine);
};

//kernel functions for image resampling with three interpolation variations
class CPUResampleImageKernel : public ResampleImageKernel {
public:
	CPUResampleImageKernel(Context* con, std::string name, const Platform& platform) : ResampleImageKernel(con->getCurrentFloating(), con->getCurrenbtWarped(), con->getCurrentDeformationField(), con->getCurrentReferenceMask(), name, platform) {
	}

	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

