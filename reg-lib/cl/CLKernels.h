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
	CLAffineDeformationFieldKernel(Context* con, std::string nameIn, const Platform& platformIn) : AffineDeformationFieldKernel(con->getCurrentDeformationField(), con->getTransformationMatrix(), nameIn, platformIn, con->getCurrentReferenceMask()) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute( bool compose = false);

};
//Kernel functions for block matching
class CLBlockMatchingKernel : public BlockMatchingKernel {
public:

	CLBlockMatchingKernel(Context* con, std::string name, const Platform& platform) : BlockMatchingKernel(con->getCurrentReference(), con->getCurrentWarped(), con->getBlockMatchingParams(), con->getCurrentReferenceMask(), name, platform) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute();


};
//a kernel function for convolution (gaussian smoothing?)
class CLConvolutionKernel : public ConvolutionKernel {
public:

	CLConvolutionKernel(std::string name, const Platform& platform) : ConvolutionKernel(name, platform) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL){}
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) {}

};

//kernel functions for numerical optimisation
class CLOptimiseKernel : public OptimiseKernel {
public:

	CLOptimiseKernel(Context* con, std::string name, const Platform& platform) : OptimiseKernel(con->getBlockMatchingParams(), con->getTransformationMatrix(),name, platform) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute(bool affine) {}
};

//kernel functions for image resampling with three interpolation variations
class CLResampleImageKernel : public ResampleImageKernel {
public:
	CLResampleImageKernel(Context* con, std::string name, const Platform& platform) : ResampleImageKernel(con->getCurrentFloating(), con->getCurrenbtWarped(), con->getCurrentDeformationField(), con->getCurrentReferenceMask(),name, platform) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;

	void execute( int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

