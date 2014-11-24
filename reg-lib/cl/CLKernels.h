#ifndef CLPKERNELS_H
#define CLPKERNELS_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "Kernels.h"


class Context;
class ClContext;
class CLContextSingletton;

//Kernel functions for affine deformation field
class CLAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
	CLAffineDeformationFieldKernel(Context* conIn, std::string nameIn);
	~CLAffineDeformationFieldKernel();

	void execute(bool compose = false);

	mat44 *affineTransformation, *targetMatrix;
	nifti_image *deformationFieldImage;
	ClContext* con;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_context clContext;
	cl_program program;

	cl_mem clDeformationField, clMask;
	CLContextSingletton* sContext;


};
//Kernel functions for block matching
class CLBlockMatchingKernel: public BlockMatchingKernel {
public:

	CLBlockMatchingKernel(Context* conIn, std::string name);
	~CLBlockMatchingKernel();
	void execute();

	CLContextSingletton* sContext;
	ClContext* con;
	nifti_image* target;
	_reg_blockMatchingParam* params;

	cl_kernel kernel;
	cl_context clContext;
	cl_program program;
	cl_command_queue commandQueue;

	cl_mem activeBlock, targetImageArray, resultImageArray, resultPosition, targetPosition, mask, targetMat;

};
//a kernel function for convolution (gaussian smoothing?)
class CLConvolutionKernel: public ConvolutionKernel {
public:

	CLConvolutionKernel(std::string name);
	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
	CLContextSingletton* sContext;
};

//kernel functions for numerical optimisation
class CLOptimiseKernel: public OptimiseKernel {
public:

	CLOptimiseKernel(Context* con, std::string name);
	void execute(bool affine);
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CLContextSingletton* sContext;
	ClContext* con;
};

//kernel functions for image resampling with three interpolation variations
class CLResampleImageKernel: public ResampleImageKernel {
public:

	CLResampleImageKernel(Context* conIn, std::string name);
	~CLResampleImageKernel();

	nifti_image *floatingImage;
	nifti_image *warpedImage;
	int *mask;
	CLContextSingletton* sContext;

	ClContext* con;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_context clContext;
	cl_program program;

	cl_mem clCurrentFloating, clCurrentDeformationField, clCurrentWarped, clMask, floMat;

	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

#endif
