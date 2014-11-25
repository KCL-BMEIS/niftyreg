#include <iostream>
#include "nifti1_io.h"

#include "Context.h"
#include "CLContextSingletton.h"
#include "CLContext.h"
#include "config.h"

#include "CLKernels.h"
#include "_reg_tools.h"
#include"_reg_resampling.h"

#include <cstring>

#define SIZE 128
#define BLOCK_SIZE 64

unsigned int min_cl(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}
//===========================================================
CLConvolutionKernel::CLConvolutionKernel(std::string name) :
		ConvolutionKernel(name) {
	sContext = &CLContextSingletton::Instance();
}
void CLConvolutionKernel::execute(nifti_image *image, float *sigma,
		int kernelType, int *mask, bool *timePoints, bool *axis) {
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints,
			axis);
}
//==========================================================
//==============================Affine Kernel CL===================================================
CLAffineDeformationFieldKernel::CLAffineDeformationFieldKernel(Context* conIn,
		std::string nameIn) :
		AffineDeformationFieldKernel(nameIn) {
	con = (ClContext*) conIn;
	sContext = &CLContextSingletton::Instance();
	this->deformationFieldImage = con->CurrentDeformationField;
	this->affineTransformation = con->getTransformationMatrix();
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

	cl_uint errNum = clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting clMask.");

}
CLAffineDeformationFieldKernel::~CLAffineDeformationFieldKernel() {

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);
}

void CLAffineDeformationFieldKernel::execute(bool compose) {
//	std::cout << "CLAffineDeformationFieldKernel exec" << std::endl;
	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks =
			((this->deformationFieldImage->nx % xThreads) == 0) ?
																					(this->deformationFieldImage->nx / xThreads) :
																					(this->deformationFieldImage->nx / xThreads) + 1;
	const unsigned int yBlocks =
			((this->deformationFieldImage->ny % yThreads) == 0) ?
																					(this->deformationFieldImage->ny / yThreads) :
																					(this->deformationFieldImage->ny / yThreads) + 1;
	const unsigned int zBlocks =
			((this->deformationFieldImage->nz % zThreads) == 0) ?
																					(this->deformationFieldImage->nz / zThreads) :
																					(this->deformationFieldImage->nz / zThreads) + 1;

	mat44 transformationMatrix =
			(compose == true) ?
										*this->affineTransformation :
										reg_mat44_mul(this->affineTransformation, targetMatrix);

	float* trans = (float *) malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_int errNum;

	cl_uint3 pms_d = { this->deformationFieldImage->nx,
			this->deformationFieldImage->ny, this->deformationFieldImage->nz };

	cl_mem cltransMat = clCreateBuffer(this->clContext,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans,
			&errNum);

	cl_uint composition = compose;
	errNum = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &cltransMat);
	sContext->checkErrNum(errNum, "Error setting cltransMat.");

	errNum |= clSetKernelArg(this->kernel, 3, sizeof(cl_uint3), &pms_d);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(this->kernel, 4, sizeof(cl_uint), &composition);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	errNum |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->clDeformationField);
	sContext->checkErrNum(errNum, "Error setting clDeformationField.");

	const cl_uint dims = 3;

	const size_t globalWorkSize[dims] = { xBlocks * xThreads, yBlocks
			* yThreads, zBlocks * zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };

	errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, dims, NULL,
			globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");
	clFinish(commandQueue);
	free(trans);
	clReleaseMemObject(cltransMat);

	return;
}
//==========================================================================================================
CLResampleImageKernel::CLResampleImageKernel(Context* conIn, std::string name) :
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
	floMat = con->getFloMatClmem();

}

CLResampleImageKernel::~CLResampleImageKernel() {
	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);
}
void CLResampleImageKernel::execute(int interp, float paddingValue,
		bool *dti_timepoint, mat33 * jacMat) {

	cl_int errNum;
	// Define the DTI indices if required
	int dtiIndeces[6];
	for (int i = 0; i < 6; ++i)
		dtiIndeces[i] = -1;
	if (dti_timepoint != NULL) {

		if (jacMat == NULL) {
			printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
			printf(
					"[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
			reg_exit(1);
		}
		int j = 0;
		for (int i = 0; i < floatingImage->nt; ++i) {
			if (dti_timepoint[i] == true)
				dtiIndeces[j++] = i;
		}
		if ((floatingImage->nz > 1 && j != 6)
				&& (floatingImage->nz == 1 && j != 3)) {
			printf(
					"[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf(
					"[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	//TODO Pre-processing kernel

	// Create OpenCL kernel
	if (interp == 3)
		kernel = clCreateKernel(program, "CubicSplineResampleImage3D", NULL);
	else if (interp == 0)
		kernel = clCreateKernel(program, "NearestNeighborResampleImage", NULL);
	else
		kernel = clCreateKernel(program, "TrilinearResampleImage2", NULL);
	if (kernel == NULL) {
		std::cerr << "Failed to create kernel" << std::endl;
		return;
	}

	long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny
			* warpedImage->nz;
	const unsigned int maxThreads = sContext->getMaxThreads();
	const unsigned int maxBlocks = sContext->getMaxBlocks();

	unsigned int blocks =
			(targetVoxelNumber % maxThreads) ?
															(targetVoxelNumber / maxThreads) + 1 :
															targetVoxelNumber / maxThreads;
	blocks = min_cl(blocks, maxBlocks);

	const cl_uint dims = 1;
	const size_t globalWorkSize[dims] = { blocks * maxThreads };
	const size_t localWorkSize[dims] = { maxThreads };

	int numMats = 0; //needs to be a parameter
	float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

	cl_long2 voxelNumber = { warpedImage->nx * warpedImage->ny
			* warpedImage->nz, floatingImage->nx * floatingImage->ny
			* floatingImage->nz };
	cl_uint3 fi_xyz =
			{ floatingImage->nx, floatingImage->ny, floatingImage->nz };
	cl_uint2 wi_tu = { warpedImage->nt, warpedImage->nu };

	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem),
			&this->clCurrentFloating);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem),
			&this->clCurrentDeformationField);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clCurrentWarped);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->floMat);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 8, sizeof(float), &paddingValue);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL,
			globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");

	clFinish(commandQueue);

	//TODO Post-processing kernel

}
//==========================================================================
CLBlockMatchingKernel::CLBlockMatchingKernel(Context* conIn, std::string name) :
		BlockMatchingKernel(name) {
//	std::cout << "CLBlockMatchingKernel" << std::endl;

	sContext = &CLContextSingletton::Instance();

	con = (ClContext*) conIn;
	target = con->CurrentReference;
	params = con->blockMatchingParams;

	std::string clInstallPath(CL_KERNELS_PATH);
	std::string clKernel("blockMatchingKernel.cl");

	clContext = sContext->getContext();
	program = sContext->CreateProgram((clInstallPath + clKernel).c_str());
	commandQueue = sContext->getCommandQueue();
	// Create OpenCL kernel
	cl_int errNum;
	kernel = clCreateKernel(program, "blockMatchingKernel3", &errNum);
	sContext->checkErrNum(errNum, "Error setting bm kernel.");

	activeBlock = con->getActiveBlockClmem();
	targetImageArray = con->getReferenceImageArrayClmem();
	resultImageArray = con->getWarpedImageClmem();
	resultPosition = con->getResultPositionClmem();
	targetPosition = con->getTargetPositionClmem();
	mask = con->getMaskClmem();
	targetMat = con->getRefMatClmem();

}
CLBlockMatchingKernel::~CLBlockMatchingKernel() {
	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);
}
void CLBlockMatchingKernel::execute() {
//	std::cout << "CLBlockMatchingKernel exec" << std::endl;
	cl_int errNum;
	// Copy some required parameters over to the device

	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	cl_mem definedBlock = clCreateBuffer(this->clContext,
			CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int),
			definedBlock_h, &errNum);
	sContext->checkErrNum(errNum, "Error setting defblock.");

	cl_uint3 imageSize =
			{ this->target->nx, this->target->ny, this->target->nz };

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->resultImageArray);
	sContext->checkErrNum(errNum, "Error setting resultImageArray.");
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->targetImageArray);
	sContext->checkErrNum(errNum, "Error setting targetImageArray.");
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->resultPosition);
	sContext->checkErrNum(errNum, "Error setting resultPosition.");
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->targetPosition);
	sContext->checkErrNum(errNum, "Error setting targetPosition.");
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->activeBlock);
	sContext->checkErrNum(errNum, "Error setting mask.");
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->mask);
	sContext->checkErrNum(errNum, "Error setting mask.");
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->targetMat);
	sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &definedBlock);
	sContext->checkErrNum(errNum, "Error setting definedBlock.");
	errNum |= clSetKernelArg(kernel, 8, sizeof(cl_uint3), &imageSize);
	sContext->checkErrNum(errNum, "Error setting image size.");



	const cl_uint dims = 3;
	const size_t globalWorkSize[dims] = { params->blockNumber[0] * 4,
			params->blockNumber[1] * 4, params->blockNumber[2] * 4 };
	const size_t localWorkSize[dims] = { 4, 4, 4 };

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL,
			globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum,
			"Error queuing blockmatching kernel for execution: ");
	clFinish(commandQueue);

	errNum = clEnqueueReadBuffer(this->commandQueue, definedBlock, CL_TRUE, 0,
			sizeof(unsigned int), definedBlock_h, 0, NULL, NULL);
	params->definedActiveBlock = *definedBlock_h;
	sContext->checkErrNum(errNum, "Error reading  var after for execution: ");
	params->definedActiveBlock = *definedBlock_h;

	free(definedBlock_h);
	clReleaseMemObject(definedBlock);

}
//===========================
CLOptimiseKernel::CLOptimiseKernel(Context* conIn, std::string name) :
		OptimiseKernel(name) {

	con = (ClContext*) conIn;
	sContext = &CLContextSingletton::Instance();
	transformationMatrix = con->getTransformationMatrix();
	blockMatchingParams = con->blockMatchingParams;
}
void CLOptimiseKernel::execute(bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
//==============================

