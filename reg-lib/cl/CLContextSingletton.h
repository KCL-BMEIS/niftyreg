#ifndef CLPCONTEXTSINGLETTON_H
#define CLPCONTEXTSINGLETTON_H


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "_reg_maths.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


// Declaration
class CLContextSingletton
{
public:

	static CLContextSingletton& Instance()
	{
		static CLContextSingletton instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}

	void queryGridDims();
	void CreateContext();
	void checDebugKernelInfo(cl_program program, cl_device_id devIdIn, char* message);
	void CreateCommandQueue();
	void init();
	cl_kernel dummyKernel(cl_device_id deviceIdIn);
	void setClIdx(int clIdxIn);

	cl_program CreateProgram( const char* fileName);


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
    bool getIsCardDoubleCapable();

	size_t getwarpGroupLength(cl_kernel kernel);

private:
	static CLContextSingletton* _instance;

	CLContextSingletton();
    ~CLContextSingletton() {
        shutDown();
	}

	CLContextSingletton(CLContextSingletton const&);// Don't Implement
	void operator=(CLContextSingletton const&); // Don't implement

	void pickCard(cl_uint deviceId);

	cl_context context;
	cl_device_id deviceId;
	cl_device_id *devices;
	cl_command_queue commandQueue;
	cl_uint numPlatforms;
	cl_platform_id* platformIds;
	cl_uint  numDevices;
	size_t maxThreads;

    bool isCardDoubleCapable;
	unsigned int maxBlocks;
	unsigned clIdx;
};
#endif
