#include <iostream>
#include "nifti1_io.h"

#include "CLKernels.h"
#include "_reg_tools.h"
#include"_reg_resampling.h"

#include <cstring>

#define SIZE 128
#define BLOCK_SIZE 64

unsigned int min_cl(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}

void CLAffineDeformationFieldKernel::execute(bool compose) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((this->deformationFieldImage->nx % xThreads) == 0) ? (this->deformationFieldImage->nx / xThreads) : (this->deformationFieldImage->nx / xThreads) + 1;
	const unsigned int yBlocks = ((this->deformationFieldImage->ny % yThreads) == 0) ? (this->deformationFieldImage->ny / yThreads) : (this->deformationFieldImage->ny / yThreads) + 1;
	const unsigned int zBlocks = ((this->deformationFieldImage->nz % zThreads) == 0) ? (this->deformationFieldImage->nz / zThreads) : (this->deformationFieldImage->nz / zThreads) + 1;

	mat44 transformationMatrix = (compose == true) ? *this->affineTransformation : reg_mat44_mul(this->affineTransformation, targetMatrix);

	float* trans = (float *) malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_int errNum;
	cl_ulong nxyz = this->deformationFieldImage->nx * this->deformationFieldImage->ny * this->deformationFieldImage->nz;

	cl_mem cltransMat = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
	cl_uint3 pms_d = { this->deformationFieldImage->nx, this->deformationFieldImage->ny, this->deformationFieldImage->nz };
	cl_uint composition = compose;

	errNum = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &cltransMat);
	errNum |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->clDeformationField);
	errNum |= clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &clMask);
	errNum |= clSetKernelArg(this->kernel, 3, sizeof(cl_uint3), &pms_d);
	errNum |= clSetKernelArg(this->kernel, 4, sizeof(cl_ulong), &nxyz);
	errNum |= clSetKernelArg(this->kernel, 5, sizeof(cl_uint), &composition);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	const cl_uint dims = 3;

	const size_t globalWorkSize[dims] = { xBlocks * xThreads, yBlocks * yThreads, zBlocks * zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };

	errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");

	clFinish(commandQueue);
	return;
}

void CLResampleImageKernel::execute(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {

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

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clCurrentFloating);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clCurrentDeformationField);
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

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");

	clFinish(commandQueue);

	//TODO Post-processing kernel

}

void CLBlockMatchingKernel::execute() {
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cl  block matching kernel!" << std::endl;

	cl_int errNum;
	// Copy some required parameters over to the device
	cl_uint3 imageSize = { this->target->nx, this->target->ny, this->target->nz };
	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	cl_mem definedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int), definedBlock_h, &errNum);

	const cl_uint dims = 3;
	const size_t globalWorkSize[dims] = { params->blockNumber[0] * 64, params->blockNumber[1] * 64, params->blockNumber[2] * 64 };
	const size_t localWorkSize[dims] = { 64, 1, 1 };

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->resultImageArray);
	sContext->checkErrNum(errNum, "Error setting resultImageArray.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->targetImageArray);
	sContext->checkErrNum(errNum, "Error setting targetImageArray.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->resultPosition);
	sContext->checkErrNum(errNum, "Error setting resultPosition.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->targetPosition);
	sContext->checkErrNum(errNum, "Error setting targetPosition.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->mask);
	sContext->checkErrNum(errNum, "Error setting mask.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->targetMat);
	sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &definedBlock);
	sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_uint3), &imageSize);
	sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");

	clFinish(commandQueue);

	errNum = clEnqueueReadBuffer(this->commandQueue, definedBlock, CL_TRUE, 0,  sizeof(unsigned int), definedBlock_h, 0, NULL, NULL);	params->definedActiveBlock = *definedBlock_h;
	params->definedActiveBlock = *definedBlock_h;

	free(definedBlock_h);
	clReleaseMemObject(definedBlock);

	std::cout << "===================================================" << std::endl;
}

