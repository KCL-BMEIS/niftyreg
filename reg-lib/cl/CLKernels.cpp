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

//debugging
#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"
//---

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
void CLConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
CLConvolutionKernel::~CLConvolutionKernel() {

}
//==========================================================
//==============================Affine Kernel CL===================================================
CLAffineDeformationFieldKernel::CLAffineDeformationFieldKernel(Context* conIn, std::string nameIn) :
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

void CLAffineDeformationFieldKernel::compare(bool compose) {

	nifti_image* gpuField = con->getCurrentDeformationField();
	float* gpuData = static_cast<float*>(gpuField->data);

	nifti_image *cpuField = nifti_copy_nim_info(gpuField);
	cpuField->data = (void *) malloc(gpuField->nvox * gpuField->nbyper);

	reg_affine_getDeformationField(con->transformationMatrix, cpuField, compose, con->CurrentReferenceMask);
	float*cpuData = static_cast<float*>(cpuField->data);

	int count = 0;
	float threshold = 0.000015f;

	for (unsigned long i = 0; i < gpuField->nvox; i++) {
		float base = fabs(cpuData[i])>1?fabs(cpuData[i]):fabs(cpuData[i])+1;
		if (fabs(cpuData[i] - gpuData[i])/base > threshold) {
//			printf("i: %d | cpu: %f | gpu: %f\n",i, cpuData[i], gpuData[i]);
			count++;
		}
	}

	std::cout << count << " targets have no match" << std::endl;
	if (count > 0)
		std::cin.get();
}

void CLAffineDeformationFieldKernel::calculate(bool compose) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((this->deformationFieldImage->nx % xThreads) == 0) ? (this->deformationFieldImage->nx / xThreads) : (this->deformationFieldImage->nx / xThreads) + 1;
	const unsigned int yBlocks = ((this->deformationFieldImage->ny % yThreads) == 0) ? (this->deformationFieldImage->ny / yThreads) : (this->deformationFieldImage->ny / yThreads) + 1;
	const unsigned int zBlocks = ((this->deformationFieldImage->nz % zThreads) == 0) ? (this->deformationFieldImage->nz / zThreads) : (this->deformationFieldImage->nz / zThreads) + 1;
	const cl_uint dims = 3;
	const size_t globalWorkSize[dims] = { xBlocks * xThreads, yBlocks * yThreads, zBlocks * zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };

	mat44 transformationMatrix = (compose == true) ? *this->affineTransformation : reg_mat44_mul(this->affineTransformation, targetMatrix);

	float* trans = (float *) malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_int errNum;

	cl_uint3 pms_d = { this->deformationFieldImage->nx, this->deformationFieldImage->ny, this->deformationFieldImage->nz };

	cl_mem cltransMat = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
//	clEnqueueWriteBuffer(this->commandQueue, cltransMat, CL_TRUE, 0, sizeof(float) * 16, trans, 0, NULL, NULL);

	cl_uint composition = compose;
	errNum = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &cltransMat);
	sContext->checkErrNum(errNum, "Error setting cltransMat.");
	errNum |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->clDeformationField);
	sContext->checkErrNum(errNum, "Error setting clDeformationField.");
	errNum |= clSetKernelArg(this->kernel, 3, sizeof(cl_uint3), &pms_d);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(this->kernel, 4, sizeof(cl_uint), &composition);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");
	clFinish(commandQueue);
	free(trans);
	clReleaseMemObject(cltransMat);
#ifndef NDEBUG
	compare(compose);
#endif

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
void CLResampleImageKernel::compare(int interp, float paddingValue) {
	nifti_image* def = con->getCurrentDeformationField();
	nifti_image* cWar = con->CurrentWarped;
	reg_resampleImage(con->CurrentFloating, cWar, def, con->CurrentReferenceMask, interp, paddingValue, NULL, NULL);
	float* cpuData2 = static_cast<float*>(cWar->data);
	float* cpuData = (float*) malloc(cWar->nvox * sizeof(float));
	for (int i = 0; i < cWar->nvox; ++i) {
		cpuData[i] = cpuData2[i];
	}
	nifti_image* gWar = con->getCurrentWarped(16);
	int count = 0;

	float* gpuData = static_cast<float*>(gWar->data);

	const float threshold = 0.000010;
	for (unsigned long i = 0; i < cWar->nvox; i++) {
		if (abs(cpuData[i] - gpuData[i]) > threshold) {
//			printf("i: %d | cpu: %f | gpu: %f\n", i, cpuData[i], gpuData[i]);
			count++;
		}
	}

	std::cout << count << "Resample: targets have no match" << std::endl;
	if (count > 0)
		exit(0);

}
void CLResampleImageKernel::calculate(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {

	cl_int errNum;
	// Define the DTI indices if required
	int dtiIndeces[6];
	for (int i = 0; i < 6; ++i)
		dtiIndeces[i] = -1;
	if (dti_timepoint != NULL) {

		if (jacMat == NULL) {
			printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
			reg_exit(1);
		}
		int j = 0;
		for (int i = 0; i < floatingImage->nt; ++i) {
			if (dti_timepoint[i] == true)
				dtiIndeces[j++] = i;
		}
		if ((floatingImage->nz > 1 && j != 6) && (floatingImage->nz == 1 && j != 3)) {
			printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	kernel = clCreateKernel(program, "ResampleImage3D", &errNum);
	sContext->checkErrNum(errNum, "Error setting kernel ResampleImage3D.");

	long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny * warpedImage->nz;
	const unsigned int maxThreads = sContext->getMaxThreads();
	const unsigned int maxBlocks = sContext->getMaxBlocks();

	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min_cl(blocks, maxBlocks);

	const cl_uint dims = 1;
	const size_t globalWorkSize[dims] = { blocks * maxThreads };
	const size_t localWorkSize[dims] = { maxThreads };

	int numMats = 0; //needs to be a parameter
	float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

	cl_long2 voxelNumber = { warpedImage->nx * warpedImage->ny * warpedImage->nz, floatingImage->nx * floatingImage->ny * floatingImage->nz };
	cl_uint3 fi_xyz = { floatingImage->nx, floatingImage->ny, floatingImage->nz };
	cl_uint2 wi_tu = { warpedImage->nt, warpedImage->nu };

	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);
	int datatype = con->getFloatingDatatype();

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clCurrentFloating);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 0.");
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clCurrentDeformationField);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 1.");
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clCurrentWarped);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 2.");
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 3.");
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->floMat);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 4.");
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 5.");
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 6.");
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 7.");
	errNum |= clSetKernelArg(kernel, 8, sizeof(float), &paddingValue);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 8.");
	errNum |= clSetKernelArg(kernel, 9, sizeof(cl_int), &interp);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 9.");
	errNum |= clSetKernelArg(kernel, 10, sizeof(cl_int), &datatype);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 10.");

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing interp kernel for execution: ");

	clFinish(commandQueue);
//	compare(interp, paddingValue);
	//TODO Post-processing kernel

}
void CLBlockMatchingKernel::compare() {
	nifti_image* referenceImage = con->CurrentReference;
	nifti_image* warpedImage = con->getCurrentWarped(16);
	int* mask = con->getCurrentReferenceMask();
	_reg_blockMatchingParam *refParams = con->getBlockMatchingParams();
	_reg_blockMatchingParam *cpu = new _reg_blockMatchingParam();
	initialise_block_matching_method(referenceImage, cpu, 50, 50, 1, mask, false);
	block_matching_method(referenceImage, warpedImage, cpu, mask);

	int count = 0;
	float* cpuTargetData = static_cast<float*>(cpu->targetPosition);
	float* cpuResultData = static_cast<float*>(cpu->resultPosition);

	float* cudaTargetData = static_cast<float*>(refParams->targetPosition);
	float* cudaResultData = static_cast<float*>(refParams->resultPosition);

	double maxTargetDiff = 0.0;
	double maxResultDiff = 0.0;

	double targetSum[3] ={ 0.0, 0.0, 0.0 };
	double resultSum[3] ={ 0.0, 0.0, 0.0 };


	int count2=0;

	for (unsigned long i = 0; i < refParams->definedActiveBlock; i++) {

		float cpuTargetPt[3] = { cpuTargetData[3 * i + 0], cpuTargetData[3 * i + 1], cpuTargetData[3 * i + 2] };
		float cpuResultPt[3] = { cpuResultData[3 * i + 0], cpuResultData[3 * i + 1], cpuResultData[3 * i + 2] };

		bool found = false;
		for (unsigned long j = 0; j < refParams->definedActiveBlock; j++) {
			float cudaTargetPt[3] = { cudaTargetData[3 * j + 0], cudaTargetData[3 * j + 1], cudaTargetData[3 * j + 2] };
			float cudaResultPt[3] = { cudaResultData[3 * j + 0], cudaResultData[3 * j + 1], cudaResultData[3 * j + 2] };

			targetSum[0] = fabs(cpuTargetPt[0] - cudaTargetPt[0]);
			targetSum[1] = fabs(cpuTargetPt[1] - cudaTargetPt[1]);
			targetSum[2] = fabs(cpuTargetPt[2] - cudaTargetPt[2]);

			const float threshold = 0.00001f;
			if (targetSum[0] <= threshold && targetSum[1] <= threshold && targetSum[2] <= threshold) {

				resultSum[0] = fabs(cpuResultPt[0] - cudaResultPt[0]);
				resultSum[1] = fabs(cpuResultPt[1] - cudaResultPt[1]);
				resultSum[2] = fabs(cpuResultPt[2] - cudaResultPt[2]);
				found = true;
				if (resultSum[0] > threshold || resultSum[1] > threshold|| resultSum[2] > threshold){
					printf("i: %lu | j: %lu | (dif: %f-%f-%f) | (out: %f, %f, %f) | (ref: %f, %f, %f)\n", i, j, resultSum[0], resultSum[1], resultSum[2], cpuResultPt[0], cpuResultPt[1], cpuResultPt[2], cudaResultPt[0], cudaResultPt[1], cudaResultPt[2]);
					count2++;
				}
			}
		}
		if (!found) {
			mat44 mat = referenceImage->qto_ijk;
			float out[3];
			reg_mat44_mul(&mat, cpuTargetPt, out);
			printf("i: %lu has no match | target: %f-%f-%f\n", i, out[0] / 4, out[1] / 4, out[2] / 4);
			count++;
		}
	}

	std::cout << count << "BM targets have no match" << std::endl;
	if (count > 0)
		exit(0);
	if (count2 > 0)
			exit(0);
}

//==========================================================================
CLBlockMatchingKernel::CLBlockMatchingKernel(Context* conIn, std::string name) :
		BlockMatchingKernel(name) {

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
void CLBlockMatchingKernel::calculate() {

	cl_int errNum;
	// Copy some required parameters over to the device

	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	cl_mem definedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int), definedBlock_h, &errNum);
	sContext->checkErrNum(errNum, "Error setting defblock.");

	cl_uint3 imageSize = { this->target->nx, this->target->ny, this->target->nz };

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
	const size_t globalWorkSize[dims] = { params->blockNumber[0] * 4, params->blockNumber[1] * 4, params->blockNumber[2] * 4 };
	const size_t localWorkSize[dims] = { 4, 4, 4 };

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution: ");
	clFinish(commandQueue);

	errNum = clEnqueueReadBuffer(this->commandQueue, definedBlock, CL_TRUE, 0, sizeof(unsigned int), definedBlock_h, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error reading  var after for execution: ");
	params->definedActiveBlock = *definedBlock_h;

	free(definedBlock_h);
	clReleaseMemObject(definedBlock);

#ifndef NDEBUG
	compare();
#endif
}
//===========================
CLOptimiseKernel::CLOptimiseKernel(Context* conIn, std::string name) :
		OptimiseKernel(name) {

	con = (ClContext*) conIn;
	sContext = &CLContextSingletton::Instance();
	transformationMatrix = con->getTransformationMatrix();
	blockMatchingParams = con->blockMatchingParams;
}
CLOptimiseKernel::~CLOptimiseKernel() {

}
void CLOptimiseKernel::calculate(bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
//==============================

