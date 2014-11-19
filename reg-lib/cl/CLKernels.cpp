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
	sContext->checkErrNum(errNum, "Error reading result buffer.");


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
		kernel = clCreateKernel(program, "TrilinearResampleImage", NULL);
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


	int numMats = 0;//needs to be a parameter

	float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
	float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

	mat44 *sourceIJKMatrix = (floatingImage->sform_code > 0) ? &(floatingImage->sto_ijk) : sourceIJKMatrix = &(floatingImage->qto_ijk);

	cl_long2 voxelNumber = { warpedImage->nx * warpedImage->ny * warpedImage->nz, floatingImage->nx * floatingImage->ny * floatingImage->nz };
	cl_uint3 fi_xyz = { floatingImage->nx, floatingImage->ny, floatingImage->nz };
	cl_uint2 wi_tu = { warpedImage->nt, warpedImage->nu };

	mat44ToCptr(*sourceIJKMatrix, sourceIJKMatrix_h);

	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);

	cl_mem clIjkMat = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), sourceIJKMatrix, &errNum);
	sContext->checkErrNum(errNum, "failed clIjkMat: ");


	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clCurrentFloating);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clCurrentDeformationField);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clCurrentWarped);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clMask);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &clIjkMat);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);

	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
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

	std::string clInstallPath(CL_KERNELS_PATH);
	std::string clKernel("blockMatchingKernel.cl");

	const unsigned int numMemObjects = 5;
	cl_int errNum;
	cl_context context = sContext->getContext();
	cl_command_queue commandQueue = sContext->getCommandQueue();
	// Create OpenCL program from affineDeformationKernel.cl kernel source
	cl_program program = sContext->CreateProgram((clInstallPath + clKernel).c_str());
	cl_kernel targetKernel = clCreateKernel(program, "process_target_blocks_gpu", NULL);
	cl_kernel resultKernel = clCreateKernel(program, "process_result_blocks_gpu", NULL);
	cl_mem memObjects[numMemObjects] = { 0, 0, 0, 0, 0 };

	// Copy the sform transformation matrix onto the device memort
	mat44 *xyz_mat = (target->sform_code > 0) ? &(target->sto_xyz) : &(target->qto_xyz);

	cl_float4 t_m_a = { xyz_mat->m[0][0], xyz_mat->m[0][1], xyz_mat->m[0][2], xyz_mat->m[0][3] };
	cl_float4 t_m_b = { xyz_mat->m[1][0], xyz_mat->m[1][1], xyz_mat->m[1][2], xyz_mat->m[1][3] };
	cl_float4 t_m_c = { xyz_mat->m[2][0], xyz_mat->m[2][1], xyz_mat->m[2][2], xyz_mat->m[2][3] };

	if (targetKernel == NULL) {
		std::cerr << "Failed to create targetKernel" << std::endl;
		sContext->Cleanup(program, targetKernel, memObjects, numMemObjects);
		return;
	}

	float *targetImageArray_d;
	float *resultImageArray_d;
	float *targetPosition_d;
	float *resultPosition_d;
	int *activeBlock_d;

	cl_uint3 bDim = { params->blockNumber[0], params->blockNumber[1], params->blockNumber[2] };
	const int numBlocks = bDim.s[0] * bDim.s[1] * bDim.s[2];

	// Image size
	cl_uint3 image_size = { target->nx, target->ny, target->nz };

	//remember to try textures

	//targetPosition_d
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, params->activeBlockNumber * 3 * sizeof(float), params->targetPosition, &errNum);
	sContext->checkErrNum(errNum, "failed memObj0: ");
	//resultPosition_d
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, params->activeBlockNumber * 3 * sizeof(float), params->resultPosition, &errNum);
	sContext->checkErrNum(errNum, "failed memObj1: ");

	//values
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, BLOCK_SIZE * numBlocks * sizeof(float), NULL, &errNum);
	sContext->checkErrNum(errNum, "failed memObj2: ");

	//targetImageArray_d
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * target->nvox, target->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj3: ");

	//resultImageArray_d
	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * result->nvox, result->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj4: ");

	//activeBlock_d
	memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBlocks * sizeof(int), mask, &errNum);
	sContext->checkErrNum(errNum, "failed memObj5: ");

	errNum = clSetKernelArg(targetKernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(targetKernel, 1, sizeof(cl_mem), &memObjects[2]);

	errNum |= clSetKernelArg(targetKernel, 2, sizeof(cl_mem), &memObjects[3]);
	errNum |= clSetKernelArg(targetKernel, 3, sizeof(cl_mem), &memObjects[5]);

	errNum |= clSetKernelArg(targetKernel, 4, sizeof(cl_uint3), &bDim);
	errNum |= clSetKernelArg(targetKernel, 5, sizeof(cl_uint3), &image_size);
	errNum |= clSetKernelArg(targetKernel, 6, sizeof(cl_float4), &t_m_a);
	errNum |= clSetKernelArg(targetKernel, 7, sizeof(cl_float4), &t_m_b);
	errNum |= clSetKernelArg(targetKernel, 8, sizeof(cl_float4), &t_m_c);

	sContext->checkErrNum(errNum, "Error setting target kernel arguments.");

	errNum = clSetKernelArg(resultKernel, 0, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(resultKernel, 1, sizeof(cl_mem), &memObjects[2]);

	errNum |= clSetKernelArg(resultKernel, 2, sizeof(cl_mem), &memObjects[4]);
	errNum |= clSetKernelArg(resultKernel, 3, sizeof(cl_mem), &memObjects[5]);

	errNum |= clSetKernelArg(resultKernel, 4, sizeof(cl_uint3), &bDim);
	errNum |= clSetKernelArg(resultKernel, 5, sizeof(cl_uint3), &image_size);
	errNum |= clSetKernelArg(resultKernel, 6, sizeof(cl_float4), &t_m_a);
	errNum |= clSetKernelArg(resultKernel, 7, sizeof(cl_float4), &t_m_b);
	errNum |= clSetKernelArg(resultKernel, 8, sizeof(cl_float4), &t_m_c);
	sContext->checkErrNum(errNum, "Error setting result kernel arguments.");

//
//
//
//	// We need to allocate some memory to keep track of overlap areas and values for blocks
//	unsigned int memSize = BLOCK_SIZE * params->activeBlockNumber;
//	printf("memsize: %d | abn: %d - %d\n", memSize, params->activeBlockNumber, numBlocks);
//
//	unsigned int Grid_block_matching = (unsigned int)ceil((float)params->activeBlockNumber / (float)NR_BLOCK->Block_target_block);
//	unsigned int Grid_block_matching_2 = 1;
//
//	// We have hit the limit in one dimension
//	if (Grid_block_matching > 65335) {
//		Grid_block_matching_2 = (unsigned int)ceil((float)Grid_block_matching / 65535.0f);
//		Grid_block_matching = 65335;
//	}
//
//	dim3 B1(NR_BLOCK->Block_target_block, 1, 1);
//	dim3 G1(Grid_block_matching, Grid_block_matching_2, 1);
//	printf("blocks: %d | threads: %d \n", Grid_block_matching, NR_BLOCK->Block_target_block);
//	// process the target blocks
//	process_target_blocks_gpu << <G1, B1 >> >(*targetPosition_d, targetValues);
//	NR_CUDA_CHECK_KERNEL(G1, B1)
//		NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
//#ifndef NDEBUG
//	printf("[NiftyReg CUDA DEBUG] process_target_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
//		cudaGetErrorString(cudaGetLastError()), G1.x, G1.y, G1.z, B1.x, B1.y, B1.z);
//#endif
//
//	//target: 512 | result: 384
//	const unsigned int targetThreads = 512;
//	const unsigned int resultThreads = 384;
//	const unsigned int maxBlocks = sContext->getMaxBlocks;
//	const unsigned int Result_block_matching = params->activeBlockNumber;
//	const unsigned int Result_block_matching_2 = (Result_block_matching > maxBlocks) ? (unsigned int)ceil((float)Result_block_matching / 65535.0f) : 1;
//	const unsigned int dims = 2;
//
//
//
//	const size_t globalWorkSize[dims] = { Result_block_matching*maxThreads };
//	const size_t localWorkSize[dims] = { targetThreads };
//
//	dim3 B2(NR_BLOCK->Block_result_block, 1, 1);
//	dim3 G2(Result_block_matching, Result_block_matching_2, 1);
//	process_result_blocks_gpu << <G2, B2 >> >(*resultPosition_d, targetValues);
//	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
//#ifndef NDEBUG
//	printf("[NiftyReg CUDA DEBUG] process_result_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
//		cudaGetErrorString(cudaGetLastError()), G2.x, G2.y, G2.z, B2.x, B2.y, B2.z);
//#endif
//	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
//	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
//	NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));
//	NR_CUDA_SAFE_CALL(cudaFree(targetValues));
//	NR_CUDA_SAFE_CALL(cudaFree(resultValues));
//
//	// We will simply call the CPU version as this step is probably
//	// not worth implementing on the GPU.
//	// device to host copy
//	int      memSize2 = params->activeBlockNumber * 3 * sizeof(float);
//	NR_CUDA_SAFE_CALL(cudaMemcpy(params->targetPosition, *targetPosition_d, memSize2, cudaMemcpyDeviceToHost));
//	NR_CUDA_SAFE_CALL(cudaMemcpy(params->resultPosition, *resultPosition_d, memSize2, cudaMemcpyDeviceToHost));
//
	std::cout << "===================================================" << std::endl;
}

