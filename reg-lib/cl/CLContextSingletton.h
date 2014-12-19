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
	void checDebugKernelInfo(cl_program program, char* message);
	void CreateCommandQueue();
	void init();

	void setClIdx(unsigned int clIdxIn){
		printf("pre id: %d\n", clIdxIn);
		clIdx=clIdxIn;
		printf("apr id: %d\n", clIdxIn);
		init();
	}

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
	cl_program getAffineProgram();
	cl_program getResampleProgram();
	cl_program getBlockMatchingProgram();
	unsigned int getMaxBlocks();
	size_t getwarpGroupLength(cl_kernel kernel);



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
	cl_program affineProgram, resampleProgram, blockMatchingProgram;
	unsigned int maxBlocks;
	unsigned int clIdx;
};
#endif
