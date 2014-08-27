#include "CLContextSingletton.h"
#include <iostream>
#include <fstream>
#include <sstream>

//// Implementation 
//CLContextSingletton* CLContextSingletton::_instance = 0;
//
//CLContextSingletton* CLContextSingletton::Instance() {
//	if (_instance == 0) {
//		std::cout << "Singletton called!"<<std::endl;
//		_instance = new CLContextSingletton();
//	}
//	return _instance;
//}

CLContextSingletton::CLContextSingletton(){
	std::cout << "Singletton called!" << std::endl;
	commandQueue = NULL;
	std::cout << "Context" << std::endl;
	CreateContext();
	
	std::cout << "queue!" << std::endl;
	CreateCommandQueue();
	std::cout << "Grid dims" << std::endl;
	queryGridDims();
}

void CLContextSingletton::queryGridDims(){
	std::size_t paramValueSize;
	//------------------------------------
	cl_int errNum = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &paramValueSize);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE");


	size_t* info = (size_t *)alloca(sizeof(size_t) * paramValueSize);
	errNum = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, NULL);
	checkErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE2");
	maxThreads = *info;
	maxBlocks = 65535;
}

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
void CLContextSingletton::CreateContext() {
	cl_int errNum;
	cl_platform_id firstPlatformId;
	context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0) {
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return ;
	}

	platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
	// First, query the total number of platforms
	errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
	checkErrNum(errNum, "Failed to find any OpenCL platforms.");

	errNum = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	checkErrNum(errNum, "Failed to find OpenCL devices.");

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0 };
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

	for (size_t i = 0; i < numPlatforms; i++)
	{
		std::cout << "num plats: " << numPlatforms << " plat:" << i << "props: " << contextProperties[i] << std::endl;
	}
	if (errNum != CL_SUCCESS) {
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return ;
		}
	}

	return;
}

///
//  Create a command queue on the first device available on the
//  context
//
void CLContextSingletton::CreateCommandQueue() {
	cl_int errNum;
	
	
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS) {
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return ;
	}

	if (deviceBufferSize <= 0) {
		std::cerr << "No devices available.";
		return ;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	checkErrNum(errNum, "Failed to get device IDs");

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
	checkErrNum(errNum, "Failed to create commandQueue for device 0");

	deviceId = devices[0];
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CLContextSingletton::CreateProgram( const char* fileName) {
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
	program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
	if (program == NULL) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS) {
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 16384 * sizeof(char), buildLog, NULL);

		std::cerr << "Error in kernel: " << buildLog << std::endl;

		clReleaseProgram(program);
		return NULL;
	}
	return program;
}

void CLContextSingletton::shutDown(){
	if (context != 0)
		clReleaseContext(context);
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	delete[] devices;
}

///
//  Cleanup any created OpenCL resources
//
void CLContextSingletton::Cleanup( cl_program program, cl_kernel kernel, cl_mem* memObjects, int length) {
	for (int i = 0; i < length; i++) {
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	

}

void CLContextSingletton::checkErrNum(cl_int errNum, std::string message) {
	if (errNum != CL_SUCCESS) {
		std::cerr << message << errNum << std::endl;

	}
}

cl_context CLContextSingletton::getContext(){
	return context;
}
cl_device_id CLContextSingletton::getDeviceId(){
	return deviceId;
}
cl_device_id* CLContextSingletton::getDevices(){
	return devices;
}
cl_command_queue CLContextSingletton::getCommandQueue(){
	return commandQueue;
}
cl_uint CLContextSingletton::getNumPlatforms(){
	return numPlatforms;
}
cl_platform_id* CLContextSingletton::getPlatformIds(){
	return platformIds;
}
cl_uint  CLContextSingletton::getNumDevices(){
	return numDevices;
}
size_t CLContextSingletton::getMaxThreads(){
	return maxThreads;
}
unsigned int CLContextSingletton::getMaxBlocks(){
	return maxBlocks;
}