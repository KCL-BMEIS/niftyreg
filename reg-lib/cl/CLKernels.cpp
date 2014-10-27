
#include <iostream>
#include "nifti1_io.h"

#include "CLKernels.h"
#include "_reg_tools.h"
#include"_reg_resampling.h"



#define SIZE 128

unsigned int min_cl(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}



void CLAffineDeformationFieldKernel::execute(bool compose) {
	std::cout << "Launching CL  affine kernel!" << std::endl;
	
	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ( ( this->deformationFieldImage->nx % xThreads ) == 0 ) ? ( this->deformationFieldImage->nx / xThreads ) : ( this->deformationFieldImage->nx / xThreads ) + 1;
	const unsigned int yBlocks = ( ( this->deformationFieldImage->ny % yThreads ) == 0 ) ? ( this->deformationFieldImage->ny / yThreads ) : ( this->deformationFieldImage->ny / yThreads ) + 1;
	const unsigned int zBlocks = ( ( this->deformationFieldImage->nz % zThreads ) == 0 ) ? ( this->deformationFieldImage->nz / zThreads ) : ( this->deformationFieldImage->nz / zThreads ) + 1;

	//inits
	int *tempMask = mask;
	if( mask == NULL ) {
		tempMask = (int *)calloc(this->deformationFieldImage->nx*
								 this->deformationFieldImage->ny*
								 this->deformationFieldImage->nz,
								 sizeof(int));
	}

	const mat44 *targetMatrix = ( this->deformationFieldImage->sform_code>0 ) ? &( this->deformationFieldImage->sto_xyz ) : &( this->deformationFieldImage->qto_xyz );
	mat44 transformationMatrix = ( compose == true ) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float* trans = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_context context = sContext->getContext();
	cl_program program = sContext->CreateProgram("C:\\Users\\thanasio\\Source\\Repos\\niftyreg-git\\reg-lib\\cl\\affineDeformationKernel.cl");
	cl_command_queue commandQueue = sContext->getCommandQueue();
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;
	
	

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "affineKernel", NULL);
	//assert(kernel != NULL);

	cl_ulong nxyz = this->deformationFieldImage->nx*this->deformationFieldImage->ny* this->deformationFieldImage->nz;

	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
	sContext->checkErrNum(errNum, "failed memObj0: ");
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * this->deformationFieldImage->nvox, this->deformationFieldImage->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj1: ");
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * nxyz, tempMask, &errNum);
	sContext->checkErrNum(errNum, "failed memObj2: ");
	
	cl_uint3 pms_d = { this->deformationFieldImage->nx,this->deformationFieldImage->ny, this->deformationFieldImage->nz };
	cl_uint composition = compose;

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint3), &pms_d);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_ulong), &nxyz);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &composition);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");





	const cl_uint dims = 3;

	const size_t globalWorkSize[dims] = { xBlocks*xThreads, yBlocks*yThreads, zBlocks*zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };


	/*std::clock_t start;

	start = std::clock();*/
	// Queue the kernel up for execution across the array
	cl_event prof_event;
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, &prof_event);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");


	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, this->deformationFieldImage->nvox * sizeof(float), this->deformationFieldImage->data, 0, NULL, NULL);
	//	std::cout << "CL Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;

	clFinish(commandQueue);
	errNum = clWaitForEvents(1, &prof_event);
	errNum |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
	errNum |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);
	float run_time_gpu = (float)( ev_end_time - ev_start_time ) / 1000000.0;
	std::cout << "CL Time: " << run_time_gpu << " ms" << std::endl;
	sContext->checkErrNum(errNum, "Error reading result buffer.");

	std::cout << "Executed program succesfully." << std::endl;
	sContext->Cleanup(program, kernel, memObjects, 3);
	//	delete (result);
	if( mask == NULL )
		free(tempMask);
	return;
}


void CLResampleImageKernel::execute( int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {
	
	
	std::cout << "running CL" << std::endl;
	if (floatingImage->datatype != warpedImage->datatype) {
		printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	if (floatingImage->nt != warpedImage->nt) {
		printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	// Define the DTI indices if required
	int dtiIndeces[6];
	for (int i = 0; i < 6; ++i) dtiIndeces[i] = -1;
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
		if ((floatingImage->nz>1 && j != 6) && (floatingImage->nz == 1 && j != 3)) {
			printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	// a mask array is created if no mask is specified
	bool MrPropreRules = false;
	if (mask == NULL) {
		// voxels in the backgreg_round are set to -1 so 0 will do the job here
		mask = (int *)calloc(warpedImage->nx*warpedImage->ny*warpedImage->nz, sizeof(int));
		MrPropreRules = true;
	}

	const unsigned int numMemObjects = 5;


	cl_int errNum;
	cl_context context = sContext->getContext();
	cl_command_queue commandQueue = sContext->getCommandQueue();
	// Create OpenCL program from resampleKernel.cl kernel source: obviously temporary, relative path needed
	cl_program program = sContext->CreateProgram("C:\\Users\\thanasio\\Source\\Repos\\niftyreg-git\\reg-lib\\cl\\resampleKernel.cl");
	cl_kernel kernel = 0;
	cl_mem memObjects[numMemObjects] = { 0, 0, 0, 0, 0 };




	// Create OpenCL kernel
	if (interp == 3)
		kernel = clCreateKernel(program, "CubicSplineResampleImage3D", NULL);
	else if (interp == 0)
		kernel = clCreateKernel(program, "NearestNeighborResampleImage", NULL);
	else
		kernel = clCreateKernel(program, "TrilinearResampleImage", NULL);
	if (kernel == NULL) {
		std::cerr << "Failed to create kernel" << std::endl;
		sContext->Cleanup(program, kernel, memObjects, numMemObjects);
		return;
	}


	long targetVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;
	const unsigned int maxThreads = sContext->getMaxThreads();
	const unsigned int maxBlocks = sContext->getMaxBlocks();

	
	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min_cl(blocks, maxBlocks);


	std::cout << "max blocks : " << maxBlocks << " max threads: " << maxThreads << std::endl;
	std::cout << " blocks : " << blocks << " threads: " << maxThreads << std::endl;

	const cl_uint dims = 1;

	const size_t globalWorkSize[dims] = { blocks*maxThreads };
	const size_t localWorkSize[dims] = { maxThreads };

	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData = NULL;
	originalFloatingData = (void *)malloc(floatingImage->nvox*sizeof(float));
	memcpy(originalFloatingData, floatingImage->data, floatingImage->nvox*sizeof(float));


	int numMats = 0;
	mat44 *sourceIJKMatrix;
	float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
	float* jacMat_h = (float*)malloc(9 * numMats*sizeof(float));

	if (floatingImage->sform_code > 0)
		sourceIJKMatrix = &(floatingImage->sto_ijk);
	else sourceIJKMatrix = &(floatingImage->qto_ijk);


	cl_long2 voxelNumber = { warpedImage->nx*warpedImage->ny*warpedImage->nz, floatingImage->nx*floatingImage->ny*floatingImage->nz };
	cl_uint3 fi_xyz = { floatingImage->nx, floatingImage->ny, floatingImage->nz };
	cl_uint2 wi_tu = { warpedImage->nt, warpedImage->nu };


	mat44ToCptr(*sourceIJKMatrix, sourceIJKMatrix_h);

	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);

	char* floating = "floating";
	char* floating1 = "deformationField_d";
	char* floating2 = "warpedImage_d";
	char* floating3 = "mask_d";
	char* floating4 = "matrix";


	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * floatingImage->nvox, floatingImage->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj0: ");



	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deformationField->nvox, deformationField->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj1: ");


	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, warpedImage->nvox * sizeof(float), warpedImage->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj2: ");


	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, floatingImage->nvox * sizeof(int), mask, &errNum);
	sContext->checkErrNum(errNum, "failed memObj3: ");

	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), sourceIJKMatrix, &errNum);
	sContext->checkErrNum(errNum, "failed memObj4: ");


	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);

	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
	errNum |= clSetKernelArg(kernel, 8, sizeof(float), &paddingValue);

	sContext->checkErrNum(errNum,"Error setting kernel arguments.");



	// Queue the kernel up for execution across the array
	cl_event prof_event;
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, &prof_event);
	sContext->checkErrNum(errNum, "Error queuing kernel for execution: ");


	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, warpedImage->nvox * sizeof(float), warpedImage->data, 0, NULL, NULL);
	//	std::cout << "CL Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;

	clFinish(commandQueue);
	errNum = clWaitForEvents(1, &prof_event);
	errNum |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
	errNum |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);
	float run_time_gpu = (float)(ev_end_time - ev_start_time) / 1000000.0;
	std::cout << "CL Time: " << run_time_gpu << " ms" << std::endl;

	sContext->checkErrNum(errNum, "Error reading result buffer.");


	std::cout << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	sContext->Cleanup(program, kernel, memObjects, 5);


	printf("done %s\n", floating);
	// The temporary logged floating array is deleted
	if (originalFloatingData != NULL) {
		free(floatingImage->data);
		floatingImage->data = originalFloatingData;
		originalFloatingData = NULL;
	}
	// The interpolated tensors are reoriented and exponentiated
	//reg_dti_resampling_postprocessing<float> << <mygrid, myblocks >> >(warpedImage_d, NULL, mask_d, jacMat_d, dtiIndeces_d, fi_xyz, wi_tu);
	reg_dti_resampling_postprocessing<float>(warpedImage, mask, jacMat, dtiIndeces);

	if (MrPropreRules == true) {
		free(mask);
		mask = NULL;
	}
}

void CLBlockMatchingKernel::execute(){
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cl  block matching kernel!" << std::endl;

	const unsigned int numMemObjects = 5;
	cl_int errNum;
	cl_context context = sContext->getContext();
	cl_command_queue commandQueue = sContext->getCommandQueue();
	// Create OpenCL program from affineDeformationKernel.cl kernel source
	cl_program program = sContext->CreateProgram("C:\\Users\\thanasio\\Source\\Repos\\niftyreg-git\\reg-lib\\cl\\blockMatchingKernel.cl");
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
	int   *activeBlock_d;

	cl_uint3 bDim = { params->blockNumber[0], params->blockNumber[1], params->blockNumber[2] };
	const int numBlocks = bDim.s[0]*bDim.s[1]*bDim.s[2];

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
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, BLOCK_SIZE *numBlocks * sizeof(float), NULL, &errNum);
	sContext->checkErrNum(errNum, "failed memObj2: ");




	//targetImageArray_d
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * target->nvox, target->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj3: ");

	//resultImageArray_d
	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * result->nvox, result->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj4: ");

	//activeBlock_d
	memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBlocks  * sizeof(int), mask, &errNum);
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

