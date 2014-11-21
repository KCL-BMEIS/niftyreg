#ifndef CLPCONTEXTSINGLETTON_H
#define CLPCONTEXTSINGLETTON_H


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

// Declaration
class CLContextSingletton {
public:

	static CLContextSingletton& Instance()
	{
		static CLContextSingletton    instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}

	void queryGridDims();

	void CreateContext();

	///
	//  Create a command queue on the first device available on the
	//  context
	//
	void CreateCommandQueue();

	///
	//  Create an OpenCL program from the kernel source file
	//
	cl_program CreateProgram( const char* fileName);

	///
	//  Cleanup any created OpenCL resources
	//
	void Cleanup(cl_program program, cl_kernel kernel, cl_mem* memObjects, int length);
	void checkErrNum(cl_int errNum, std::string message);
	void shutDown();

	cl_context getContext();
	cl_device_id getDeviceId();
	cl_device_id* getDevices();
	cl_command_queue getCommandQueue();
	cl_uint getNumPlatforms();
	cl_platform_id* getPlatformIds();
	cl_uint getNumDevices();
	size_t getMaxThreads();
	unsigned int getMaxBlocks();



private:

	static CLContextSingletton* _instance;

	CLContextSingletton();
	~CLContextSingletton(){
		shutDown();
	}

	CLContextSingletton(CLContextSingletton const&);// Don't Implement
	void operator=(CLContextSingletton const&); // Don't implement


	cl_context context;
	cl_device_id deviceId;
	cl_device_id *devices;
	cl_command_queue commandQueue;
	cl_uint numPlatforms;
	cl_platform_id* platformIds;
	cl_uint  numDevices;
	size_t maxThreads;
	unsigned int maxBlocks;
};
#endif
