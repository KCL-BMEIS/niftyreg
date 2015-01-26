#ifndef CLPKERNELS_H
#define CLPKERNELS_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "Kernels.h"


class Content;
class ClContent;
class CLContextSingletton;

//Kernel functions for affine deformation field
class CLAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
	CLAffineDeformationFieldKernel(Content* conIn, std::string nameIn);
	~CLAffineDeformationFieldKernel();

	void calculate(bool compose = false);
	void compare(bool compose);
private:
	mat44 *affineTransformation, *targetMatrix;
	nifti_image *deformationFieldImage;
	ClContent* con;
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

	CLBlockMatchingKernel(Content* conIn, std::string name);
	~CLBlockMatchingKernel();
	void compare();
	void calculate(int range);
private:
	CLContextSingletton* sContext;
	ClContent* con;
	nifti_image* target;
	_reg_blockMatchingParam* params;

	cl_kernel kernel;
	cl_context clContext;
	cl_program program;
	cl_command_queue commandQueue;

	cl_mem clActiveBlock, clTargetImageArray, clResultImageArray, clResultPosition, clTargetPosition, clMask, clTargetMat;

};
//a kernel function for convolution (gaussian smoothing?)
class CLConvolutionKernel: public ConvolutionKernel {
public:

	CLConvolutionKernel(std::string name);
	~CLConvolutionKernel();
	void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
private:
	CLContextSingletton* sContext;
};

//kernel functions for numerical optimisation
class CLOptimiseKernel: public OptimiseKernel {
public:

	CLOptimiseKernel(Content* con, std::string name);
	~CLOptimiseKernel();
	void calculate(bool affine, bool ils, bool clsvd=0);
private:
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CLContextSingletton* sContext;
	ClContent* con;
};

//kernel functions for image resampling with three interpolation variations
class CLResampleImageKernel: public ResampleImageKernel {
public:

	CLResampleImageKernel(Content* conIn, std::string name);
	~CLResampleImageKernel();

	void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
	void compare(int interp, float paddingValue);
private:

	nifti_image *floatingImage;
	nifti_image *warpedImage;
	int *mask;
	CLContextSingletton* sContext;

	ClContent* con;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_context clContext;
	cl_program program;

	cl_mem clCurrentFloating, clCurrentDeformationField, clCurrentWarped, clMask, floMat;
};

#endif
