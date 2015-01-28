#include "CLContextSingletton.h"
#include "../reg-lib/cl/InfoDevice.h"

#include "_reg_maths.h"

#include <iostream>
#include <fstream>
#include <sstream>

CLContextSingletton::CLContextSingletton() {
	this->commandQueue = NULL;
	this->context = NULL;
	this->clIdx = -1;
	init();
}
void CLContextSingletton::init() {

	// Query the number of platforms
	cl_int errNum = clGetPlatformIDs(0, NULL, &this->numPlatforms);
	checkErrNum(errNum, "Failed to find CL platforms.");

	this->platformIds = (cl_platform_id *) alloca(sizeof(cl_platform_id) * this->numPlatforms);
	errNum = clGetPlatformIDs(this->numPlatforms, this->platformIds, NULL);
	checkErrNum(errNum, "Failed to find any OpenCL platforms.");

	errNum = clGetDeviceIDs(this->platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &this->numDevices);
	checkErrNum(errNum, "Failed to find OpenCL devices.");

	this->devices = new cl_device_id[this->numDevices];
	errNum = clGetDeviceIDs(this->platformIds[0], CL_DEVICE_TYPE_ALL, this->numDevices, this->devices, NULL);

	if(clIdx<0)pickCard();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) this->platformIds[0], 0 };
	this->context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

	if (errNum != CL_SUCCESS) {
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
				NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to create an OpenCL GPU or CPU context."
					<< std::endl;
			return;
		}
	}

	this->commandQueue = clCreateCommandQueue(this->context, this->devices[this->clIdx], CL_QUEUE_PROFILING_ENABLE, NULL);
	checkErrNum(errNum, "Failed to create commandQueue for device ");

	this->deviceId = this->devices[this->clIdx];
	queryGridDims();
}
void CLContextSingletton::queryGridDims() {
	std::size_t paramValueSize;
	//------------------------------------
	cl_int errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &paramValueSize);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE");

	size_t* info = (size_t *) alloca(sizeof(size_t) * paramValueSize);
	errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, NULL);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE2");
	this->maxThreads = *info;
	this->maxBlocks = 65535;
}

void CLContextSingletton::pickCard() {
	cl_uint maxProcs = 0;
	this->clIdx = 0;
	cl_int errNum;
	std::size_t paramValueSize;


	for (int i = 0; i < this->numDevices; ++i) {
		errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &paramValueSize);
		checkErrNum(errNum, "Failed to find OpenCL device info ");
		cl_uint * info = (cl_uint *) alloca(sizeof(cl_uint) * paramValueSize);
		errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, paramValueSize, info, NULL);
		checkErrNum(errNum, "Failed to find OpenCL device info ");
		cl_uint numProcs = *info;
		const bool found = numProcs > maxProcs;
		this->clIdx = found ? i : this->clIdx;
		maxProcs = found ? numProcs : maxProcs;
	}

}

cl_program CLContextSingletton::CreateProgram(const char* fileName) {
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(this->context, 1, (const char**) &srcStr, NULL, &errNum);
	checkErrNum(errNum, "Failed to create CL program");

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS) checDebugKernelInfo(program,this->deviceId, "Errors in kernel: ");

	return program;
}

void CLContextSingletton::shutDown() {
	/*std::cout << "Shutting down cl" << std::endl;*/
	if (this->context != 0) clReleaseContext(this->context);
	if (this->commandQueue != 0) clReleaseCommandQueue(this->commandQueue);

	delete this->devices;
}

void CLContextSingletton::checDebugKernelInfo(cl_program program, cl_device_id devIdIn, char* message) {
	char buffer[10240];

	clGetProgramBuildInfo(program, devIdIn, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
	reg_print_fct_error(message);
	reg_print_fct_error(buffer);
}

void CLContextSingletton::checkErrNum(cl_int errNum, std::string message) {
	if (errNum != CL_SUCCESS) reg_print_fct_error(message.c_str());
}

cl_context CLContextSingletton::getContext() {
	return this->context;
}
cl_device_id CLContextSingletton::getDeviceId() {
	return this->deviceId;
}
cl_device_id* CLContextSingletton::getDevices() {
	return this->devices;
}
cl_command_queue CLContextSingletton::getCommandQueue() {
	return this->commandQueue;
}
cl_uint CLContextSingletton::getNumPlatforms() {
	return this->numPlatforms;
}
cl_platform_id* CLContextSingletton::getPlatformIds() {
	return this->platformIds;
}
cl_uint CLContextSingletton::getNumDevices() {
	return this->numDevices;
}
size_t CLContextSingletton::getMaxThreads() {
	return this->maxThreads;
}
unsigned int CLContextSingletton::getMaxBlocks() {
	return this->maxBlocks;
}

size_t CLContextSingletton::getwarpGroupLength(cl_kernel kernel) {
	size_t local;
	// Get the maximum work group size for executing the kernel on the device
	cl_int err = clGetKernelWorkGroupInfo(kernel, this->deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, NULL);
	checkErrNum(err, "Error: Failed to retrieve kernel work group info!");

	return local;
}
cl_kernel CLContextSingletton::dummyKernel(cl_device_id deviceIdIn) {

	const char *source = "\n"
			"__kernel void dummy(                                                       \n"
			"   __global float* in,                                              \n"
			"   __global float* out,                                             \n"
			"   const unsigned int count)                                           \n"
			"{                                                                      \n"
			"   int i = get_global_id(0);                                           \n"
			"   if(i < count)                                                       \n"
			"       out[i] = in[i] * out[i];                               			 \n"
			"}                                                                      \n"
			"\n";

	cl_int  err ;
	cl_program program = clCreateProgramWithSource(this->context, 1, (const char **) & source, NULL, &err);
	checkErrNum(err, "Failed to create CL program");
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) checDebugKernelInfo(program,deviceIdIn, "Errors in kernel: ");

	// Create the compute kernel in the program we wish to run
	//
	cl_kernel kernel = clCreateKernel(program, "dummy", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		reg_print_fct_error("Error: Failed to create compute kernel!\n");
		return NULL;
	}
	return kernel;
}
