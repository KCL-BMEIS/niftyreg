#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include "nifti1_io.h"

#include "CLKernels.h"
#include "_reg_tools.h"

#define SIZE 128
///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext() {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if( errNum != CL_SUCCESS || numPlatforms <= 0 ) {
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0 };
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	std::cout << "num plats: " << numPlatforms << " 0:" << contextProperties[0] << " 1: " << contextProperties[1] << " 2: " << contextProperties[2] << std::endl;
	if( errNum != CL_SUCCESS ) {
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if( errNum != CL_SUCCESS ) {
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if( errNum != CL_SUCCESS ) {
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if( deviceBufferSize <= 0 ) {
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if( errNum != CL_SUCCESS ) {
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
	if( commandQueue == NULL ) {
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName) {
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if( !kernelFile.is_open() ) {
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
	if( program == NULL ) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if( errNum != CL_SUCCESS ) {
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 16384 * sizeof(char), buildLog, NULL);

		std::cerr << "Error in kernel: " << buildLog << std::endl;

		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3], int length) {
	for( int i = 0; i < length; i++ ) {
		if( memObjects[i] != 0 )
			clReleaseMemObject(memObjects[i]);
	}
	if( commandQueue != 0 )
		clReleaseCommandQueue(commandQueue);

	if( kernel != 0 )
		clReleaseKernel(kernel);

	if( program != 0 )
		clReleaseProgram(program);

	if( context != 0 )
		clReleaseContext(context);

}

void checkErrNum(cl_int errNum, std::string message) {
	if( errNum != CL_SUCCESS ) {
		std::cerr << message << errNum << std::endl;

	}
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





	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;

	// Create an OpenCL context on first available platform
	context = CreateContext();
	if( context == NULL ) {
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return;
	}

	// Create a command-queue on the first device available
	// on the created context
	commandQueue = CreateCommandQueue(context, &device);
	if( commandQueue == NULL ) {
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}

	std::cout << "Program: " << std::endl;
	// Create OpenCL program from affineDeformationKernel.cl kernel source
	program = CreateProgram(context, device, "affineDeformationKernel.cl");
	if( program == NULL ) {
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "affineKernel", NULL);
	if( kernel == NULL ) {
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}
	std::cout << "Kernel: " << std::endl;
	cl_ulong nxyz = deformationField->nx*deformationField->ny* deformationField->nz;

	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
	checkErrNum(errNum, "failed memObj0: ");
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * deformationField->nvox, deformationField->data, &errNum);
	checkErrNum(errNum, "failed memObj1: ");
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * nxyz, tempMask, &errNum);
	checkErrNum(errNum, "failed memObj2: ");
	
	cl_uint3 pms_d = { deformationField->nx,deformationField->ny, deformationField->nz };
	cl_uint composition = compose;

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint3), &pms_d);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_ulong), &nxyz);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &composition);


	if( errNum != CL_SUCCESS ) {
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}



	const cl_uint dims = 3;

	const size_t globalWorkSize[dims] = { xBlocks*xThreads, yBlocks*yThreads, zBlocks*zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };


	/*std::clock_t start;

	start = std::clock();*/
	// Queue the kernel up for execution across the array
	cl_event prof_event;
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, &prof_event);
	checkErrNum(errNum, "Error queuing kernel for execution: ");
	if( errNum != CL_SUCCESS ) {
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}

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


	if( errNum != CL_SUCCESS ) {
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects, 3);
		return;
	}


	std::cout << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects, 3);
	//	delete (result);
	if( mask == NULL )
		free(tempMask);
	return;
}