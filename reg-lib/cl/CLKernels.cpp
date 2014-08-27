
#include <iostream>
#include "nifti1_io.h"

#include "CLKernels.h"
#include "_reg_tools.h"
#include"_reg_resampling.h"



#define SIZE 128

unsigned int min_cl(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}







void CLAffineDeformationFieldKernel::execute(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask) {
	std::cout << "Launching CL  affine kernel!" << std::endl;
	
	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ( ( deformationField->nx % xThreads ) == 0 ) ? ( deformationField->nx / xThreads ) : ( deformationField->nx / xThreads ) + 1;
	const unsigned int yBlocks = ( ( deformationField->ny % yThreads ) == 0 ) ? ( deformationField->ny / yThreads ) : ( deformationField->ny / yThreads ) + 1;
	const unsigned int zBlocks = ( ( deformationField->nz % zThreads ) == 0 ) ? ( deformationField->nz / zThreads ) : ( deformationField->nz / zThreads ) + 1;

	//inits
	int *tempMask = mask;
	if( mask == NULL ) {
		tempMask = (int *)calloc(deformationField->nx*
								 deformationField->ny*
								 deformationField->nz,
								 sizeof(int));
	}

	const mat44 *targetMatrix = ( deformationField->sform_code>0 ) ? &( deformationField->sto_xyz ) : &( deformationField->qto_xyz );
	mat44 transformationMatrix = ( compose == true ) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float* trans = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_context context = sContext->getContext();
	cl_program program = sContext->CreateProgram("affineDeformationKernel.cl");
	cl_command_queue commandQueue = sContext->getCommandQueue();
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;
	
	

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "affineKernel", NULL);
	assert(kernel != NULL);

	cl_ulong nxyz = deformationField->nx*deformationField->ny* deformationField->nz;

	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
	sContext->checkErrNum(errNum, "failed memObj0: ");
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * deformationField->nvox, deformationField->data, &errNum);
	sContext->checkErrNum(errNum, "failed memObj1: ");
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * nxyz, tempMask, &errNum);
	sContext->checkErrNum(errNum, "failed memObj2: ");
	
	cl_uint3 pms_d = { deformationField->nx,deformationField->ny, deformationField->nz };
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
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, deformationField->nvox * sizeof(float), deformationField->data, 0, NULL, NULL);
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


void CLResampleImageKernel::execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {
	cl_context context = sContext->getContext();
	cl_command_queue commandQueue = sContext->getCommandQueue();
	
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

	// Create OpenCL program from affineDeformationKernel.cl kernel source
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



	// The DTI are logged
	reg_dti_resampling_preprocessing<float>(floatingImage, &originalFloatingData, dtiIndeces);
	//reg_dti_resampling_preprocessing<float> << <mygrid, myblocks >> >(floatingImage_d, dtiIndeces, fi_xyz);

	printf("kernel %s\n", floating);

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

