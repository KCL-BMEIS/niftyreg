#include "CLContextSingletton.h"
#include "../reg-lib/cl/InfoDevice.h"

#include "_reg_maths.h"

#include <iostream>
#include <fstream>
#include <sstream>

/* *************************************************************** */
CLContextSingletton::CLContextSingletton()
{
	this->commandQueue = NULL;
	this->context = NULL;
	this->clIdx = -1;
	init();
}
/* *************************************************************** */
void CLContextSingletton::init()
{
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

	if(clIdx<0)
		pickCard();

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
/* *************************************************************** */
void CLContextSingletton::queryGridDims()
{
	std::size_t paramValueSize;
	cl_int errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &paramValueSize);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE");

	size_t* info = (size_t *) alloca(sizeof(size_t) * paramValueSize);
	errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, NULL);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE2");
	this->maxThreads = *info;
	this->maxBlocks = 65535;
}
/* *************************************************************** */
void CLContextSingletton::pickCard()
{
	cl_uint maxProcs = 0;
	this->clIdx = 0;
	cl_int errNum;
	std::size_t paramValueSize;
	for(cl_uint i = 0; i < this->numDevices; ++i)
	{
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
/* *************************************************************** */
cl_program CLContextSingletton::CreateProgram(const char* fileName)
{
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
	if (errNum != CL_SUCCESS) checDebugKernelInfo(program,this->deviceId, (char *)"Errors in kernel: ");

	return program;
}
/* *************************************************************** */
void CLContextSingletton::shutDown()
{
	/*std::cout << "Shutting down cl" << std::endl;*/
	if (this->context != 0) clReleaseContext(this->context);
	if (this->commandQueue != 0) clReleaseCommandQueue(this->commandQueue);

	delete this->devices;
}
/* *************************************************************** */
void CLContextSingletton::checDebugKernelInfo(cl_program program, cl_device_id devIdIn, char* message)
{
	char buffer[10240];

	clGetProgramBuildInfo(program, devIdIn, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
	reg_print_fct_error(message);
	reg_print_fct_error(buffer);
}
/* *************************************************************** */
void CLContextSingletton::checkErrNum(cl_int errNum, std::string message)
{
	if (errNum != CL_SUCCESS)
	{
		reg_print_msg_error(message.c_str());
		switch(errNum){
		case -1: reg_print_msg_error("CL_DEVICE_NOT_FOUND");break;
		case -2: reg_print_msg_error("CL_DEVICE_NOT_AVAILABLE");break;
		case -3: reg_print_msg_error("CL_COMPILER_NOT_AVAILABLE");break;
		case -4: reg_print_msg_error("CL_MEM_OBJECT_ALLOCATION_FAILURE");break;
		case -5: reg_print_msg_error("CL_OUT_OF_RESOURCES");break;
		case -6: reg_print_msg_error("CL_OUT_OF_HOST_MEMORY");break;
		case -7: reg_print_msg_error("CL_PROFILING_INFO_NOT_AVAILABLE");break;
		case -8: reg_print_msg_error("CL_MEM_COPY_OVERLAP");break;
		case -9: reg_print_msg_error("CL_IMAGE_FORMAT_MISMATCH");break;
		case -10: reg_print_msg_error("CL_IMAGE_FORMAT_NOT_SUPPORTED");break;
		case -11: reg_print_msg_error("CL_BUILD_PROGRAM_FAILURE");break;
		case -12: reg_print_msg_error("CL_MAP_FAILURE");break;
		case -13: reg_print_msg_error("CL_MISALIGNED_SUB_BUFFER_OFFSET");break;
		case -14: reg_print_msg_error("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");break;
		case -15: reg_print_msg_error("CL_COMPILE_PROGRAM_FAILURE");break;
		case -16: reg_print_msg_error("CL_LINKER_NOT_AVAILABLE");break;
		case -17: reg_print_msg_error("CL_LINK_PROGRAM_FAILURE");break;
		case -18: reg_print_msg_error("CL_DEVICE_PARTITION_FAILED");break;
		case -19: reg_print_msg_error("CL_KERNEL_ARG_INFO_NOT_AVAILABLE");break;
		case -30: reg_print_msg_error("CL_INVALID_VALUE");break;
		case -31: reg_print_msg_error("CL_INVALID_DEVICE_TYPE");break;
		case -32: reg_print_msg_error("CL_INVALID_PLATFORM");break;
		case -33: reg_print_msg_error("CL_INVALID_DEVICE");break;
		case -34: reg_print_msg_error("CL_INVALID_CONTEXT");break;
		case -35: reg_print_msg_error("CL_INVALID_QUEUE_PROPERTIES");break;
		case -36: reg_print_msg_error("CL_INVALID_COMMAND_QUEUE");break;
		case -37: reg_print_msg_error("CL_INVALID_HOST_PTR");break;
		case -38: reg_print_msg_error("CL_INVALID_MEM_OBJECT");break;
		case -39: reg_print_msg_error("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");break;
		case -40: reg_print_msg_error("CL_INVALID_IMAGE_SIZE");break;
		case -41: reg_print_msg_error("CL_INVALID_SAMPLER");break;
		case -42: reg_print_msg_error("CL_INVALID_BINARY");break;
		case -43: reg_print_msg_error("CL_INVALID_BUILD_OPTIONS");break;
		case -44: reg_print_msg_error("CL_INVALID_PROGRAM");break;
		case -45: reg_print_msg_error("CL_INVALID_PROGRAM_EXECUTABLE");break;
		case -46: reg_print_msg_error("CL_INVALID_KERNEL_NAME");break;
		case -47: reg_print_msg_error("CL_INVALID_KERNEL_DEFINITION");break;
		case -48: reg_print_msg_error("CL_INVALID_KERNEL");break;
		case -49: reg_print_msg_error("CL_INVALID_ARG_INDEX");break;
		case -50: reg_print_msg_error("CL_INVALID_ARG_VALUE");break;
		case -51: reg_print_msg_error("CL_INVALID_ARG_SIZE");break;
		case -52: reg_print_msg_error("CL_INVALID_KERNEL_ARGS");break;
		case -53: reg_print_msg_error("CL_INVALID_WORK_DIMENSION");break;
		case -54: reg_print_msg_error("CL_INVALID_WORK_GROUP_SIZE");break;
		case -55: reg_print_msg_error("CL_INVALID_WORK_ITEM_SIZE");break;
		case -56: reg_print_msg_error("CL_INVALID_GLOBAL_OFFSET");break;
		case -57: reg_print_msg_error("CL_INVALID_EVENT_WAIT_LIST");break;
		case -58: reg_print_msg_error("CL_INVALID_EVENT");break;
		case -59: reg_print_msg_error("CL_INVALID_OPERATION");break;
		case -60: reg_print_msg_error("CL_INVALID_GL_OBJECT");break;
		case -61: reg_print_msg_error("CL_INVALID_BUFFER_SIZE");break;
		case -62: reg_print_msg_error("CL_INVALID_MIP_LEVEL");break;
		case -63: reg_print_msg_error("CL_INVALID_GLOBAL_WORK_SIZE");break;
		case -64: reg_print_msg_error("CL_INVALID_PROPERTY");break;
		case -65: reg_print_msg_error("CL_INVALID_IMAGE_DESCRIPTOR");break;
		case -66: reg_print_msg_error("CL_INVALID_COMPILER_OPTIONS");break;
		case -67: reg_print_msg_error("CL_INVALID_LINKER_OPTIONS");break;
		case -68: reg_print_msg_error("CL_INVALID_DEVICE_PARTITION_COUNT");break;
		default : reg_print_msg_error("Unknown error type");break;
		}
		reg_exit(1);
	}
}
/* *************************************************************** */
cl_context CLContextSingletton::getContext()
{
	return this->context;
}
/* *************************************************************** */
cl_device_id CLContextSingletton::getDeviceId()
{
	return this->deviceId;
}
/* *************************************************************** */
cl_device_id* CLContextSingletton::getDevices()
{
	return this->devices;
}
/* *************************************************************** */
cl_command_queue CLContextSingletton::getCommandQueue()
{
	return this->commandQueue;
}
/* *************************************************************** */
cl_uint CLContextSingletton::getNumPlatforms()
{
	return this->numPlatforms;
}
/* *************************************************************** */
cl_platform_id* CLContextSingletton::getPlatformIds()
{
	return this->platformIds;
}
/* *************************************************************** */
cl_uint CLContextSingletton::getNumDevices()
{
	return this->numDevices;
}
/* *************************************************************** */
size_t CLContextSingletton::getMaxThreads()
{
	return this->maxThreads;
}
/* *************************************************************** */
unsigned int CLContextSingletton::getMaxBlocks()
{
	return this->maxBlocks;
}
/* *************************************************************** */
size_t CLContextSingletton::getwarpGroupLength(cl_kernel kernel)
{
	size_t local;
	// Get the maximum work group size for executing the kernel on the device
	cl_int err = clGetKernelWorkGroupInfo(kernel, this->deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, NULL);
	checkErrNum(err, "Error: Failed to retrieve kernel work group info!");

	return local;
}
/* *************************************************************** */
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
	if (err != CL_SUCCESS) checDebugKernelInfo(program,deviceIdIn, (char *)"Errors in kernel: ");

	// Create the compute kernel in the program we wish to run
	//
	cl_kernel kernel = clCreateKernel(program, "dummy", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		reg_print_fct_error("Error: Failed to create compute kernel!");
		return NULL;
	}
	return kernel;
}
/* *************************************************************** */
