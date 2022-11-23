#pragma once

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
class ClContextSingleton
{
public:

	static ClContextSingleton& Instance()
	{
		static ClContextSingleton instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}

	void queryGridDims();
	void CreateContext();
	void checDebugKernelInfo(cl_program program, cl_device_id devIdIn, char* message);
	void CreateCommandQueue();
	void init();
	cl_kernel dummyKernel(cl_device_id deviceIdIn);
	void SetClIdx(int clIdxIn);

	cl_program CreateProgram( const char* fileName);


	void Cleanup(cl_program program, cl_kernel kernel, cl_mem* memObjects, int length);
	void checkErrNum(cl_int errNum, std::string message);
	void shutDown();

	cl_context GetContext();
	cl_device_id GetDeviceId();
	cl_device_id* GetDevices();
	cl_command_queue GetCommandQueue();
	cl_uint GetNumPlatforms();
	cl_platform_id* GetPlatformIds();
	cl_uint GetNumDevices();
	size_t GetMaxThreads();

	unsigned int GetMaxBlocks();
    bool GetIsCardDoubleCapable();

	size_t GetWarpGroupLength(cl_kernel kernel);

private:
	static ClContextSingleton* _instance;

	ClContextSingleton();
    ~ClContextSingleton() {
        shutDown();
	}

	ClContextSingleton(ClContextSingleton const&);// Don't Implement
	void operator=(ClContextSingleton const&); // Don't implement

	void PickCard(cl_uint deviceId);

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
