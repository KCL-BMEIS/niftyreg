#pragma once

#include "Kernels.h"
#include "CLContextSingletton.h"
#include "CLContext.h"
#include "config.h"

class CLOptimiseKernel;
class CLAffineDeformationFieldKernel;
class CLBlockMatchingKernel;
class CLConvolutionKernel;
class CLResampleImageKernel;

//Kernel functions for affine deformation field 
class CLAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
	CLAffineDeformationFieldKernel(Context* conIn, std::string nameIn) :
			AffineDeformationFieldKernel(nameIn) {
		con = (ClContext*) conIn;
		sContext = &CLContextSingletton::Instance();
		this->deformationFieldImage = con->CurrentDeformationField;
		this->affineTransformation = con->getTransformationMatrix();
		this->mask = con->CurrentReferenceMask;

		targetMatrix = (this->deformationFieldImage->sform_code > 0) ? &(this->deformationFieldImage->sto_xyz) : &(this->deformationFieldImage->qto_xyz);

		std::string clInstallPath(CL_KERNELS_PATH);
		std::string clKernel("affineDeformationKernel.cl");

		clContext = sContext->getContext();
		program = sContext->CreateProgram((clInstallPath + clKernel).c_str());
		commandQueue = sContext->getCommandQueue();
		// Create OpenCL kernel
		kernel = clCreateKernel(program, "affineKernel", NULL);
		clDeformationField = con->getDeformationFieldArrayClmem();
		clMask = con->getMaskClmem();
	}
	~CLAffineDeformationFieldKernel() {
		std::cout << "clean up context staff" << std::endl;
		if (kernel != 0)
			clReleaseKernel(kernel);

		if (program != 0)
			clReleaseProgram(program);
	}

	void execute(bool compose = false);

	mat44 *targetMatrix;
	mat44 *affineTransformation;
	nifti_image *deformationFieldImage;
	int* mask;
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

	CLBlockMatchingKernel(Context* con, std::string name) :
			BlockMatchingKernel(name) {
		sContext = &CLContextSingletton::Instance();
		target = con->CurrentReference;
		result = con->CurrentWarped;
		params = con->blockMatchingParams;
		mask = con->CurrentReferenceMask;

		std::string clInstallPath(CL_KERNELS_PATH);
		std::string clKernel("blockMatchingKernel.cl");

		clContext = sContext->getContext();
		program = sContext->CreateProgram((clInstallPath + clKernel).c_str());
		commandQueue = sContext->getCommandQueue();
		// Create OpenCL kernel
		kernel = clCreateKernel(program, "blockMatchingKernel", NULL);
	}
	CLContextSingletton* sContext;
	void execute();

	nifti_image* target;
	nifti_image* result;
	_reg_blockMatchingParam* params;
	int* mask;

	cl_kernel kernel;
	cl_context clContext;
	cl_program program;
	cl_command_queue commandQueue;

};
//a kernel function for convolution (gaussian smoothing?)
class CLConvolutionKernel: public ConvolutionKernel {
public:

	CLConvolutionKernel(std::string name) :
			ConvolutionKernel(name) {
		sContext = &CLContextSingletton::Instance();
	}
	CLContextSingletton* sContext;
	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) {
	}
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) {
	}

};

//kernel functions for numerical optimisation
class CLOptimiseKernel: public OptimiseKernel {
public:

	CLOptimiseKernel(Context* con, std::string name) :
			OptimiseKernel(name) {
		sContext = &CLContextSingletton::Instance();
		transformationMatrix = con->getTransformationMatrix();
		blockMatchingParams = con->getBlockMatchingParams();
	}
	_reg_blockMatchingParam *blockMatchingParams;
	mat44 *transformationMatrix;
	CLContextSingletton* sContext;
	void execute(bool affine) {
	}
};

//kernel functions for image resampling with three interpolation variations
class CLResampleImageKernel: public ResampleImageKernel {
public:

	CLResampleImageKernel(Context* conIn, std::string name) :
			ResampleImageKernel(name) {

		std::string clInstallPath(CL_KERNELS_PATH);
		std::string clKernel("resampleKernel.cl");

		sContext = &CLContextSingletton::Instance();
		clContext = sContext->getContext();
		commandQueue = sContext->getCommandQueue();

		program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

		con = (ClContext*) conIn;
		floatingImage = con->CurrentFloating;
		warpedImage = con->CurrentWarped;
		mask = con->CurrentReferenceMask;

		kernel = 0;
		clCurrentFloating = con->getFloatingImageArrayClmem();
		clCurrentDeformationField = con->getDeformationFieldArrayClmem();
		clCurrentWarped = con->getWarpedImageClmem();
		clMask = con->getMaskClmem();

	}

	~CLResampleImageKernel() {
		std::cout << "CLResampleImageKernel: clean up context staff" << std::endl;
		if (kernel != 0)
			clReleaseKernel(kernel);

		if (program != 0)
			clReleaseProgram(program);
	}

	nifti_image *floatingImage;
	nifti_image *warpedImage;
	int *mask;
	CLContextSingletton* sContext;

	ClContext* con;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_context clContext;
	cl_program program;

	cl_mem clCurrentFloating, clCurrentDeformationField, clCurrentWarped, clMask;

	void execute(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

