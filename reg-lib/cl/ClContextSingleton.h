#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "_reg_tools.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


class ClContextSingleton {
public:
    ClContextSingleton(ClContextSingleton const&) = delete;
    void operator=(ClContextSingleton const&) = delete;

    static ClContextSingleton& GetInstance() {
        // Instantiated on first use.
        static ClContextSingleton instance; // Guaranteed to be destroyed.
        return instance;
    }

    cl_program CreateProgram(const char *fileName);
    void CheckErrNum(cl_int errNum, std::string message);
    cl_kernel DummyKernel(cl_device_id deviceIdIn);
    void SetClIdx(int clIdxIn);

    cl_context GetContext();
    cl_device_id GetDeviceId();
    cl_device_id* GetDevices();
    cl_command_queue GetCommandQueue();
    cl_uint GetNumPlatforms();
    cl_platform_id* GetPlatformIds();
    cl_uint GetNumDevices();
    size_t GetMaxThreads();
    unsigned GetMaxBlocks();
    size_t GetWarpGroupLength(cl_kernel kernel);
    bool IsCardDoubleCapable();

private:
    ClContextSingleton();
    ~ClContextSingleton();

    void Init();
    void PickCard(cl_uint deviceId);
    void CheckDebugKernelInfo(cl_program program, cl_device_id devIdIn, const char *message);
    void QueryGridDims();

    cl_context context;
    cl_device_id deviceId;
    cl_device_id *devices;
    cl_command_queue commandQueue;
    cl_uint numPlatforms;
    cl_platform_id *platformIds;
    cl_uint numDevices;
    size_t maxThreads;

    bool isCardDoubleCapable;
    unsigned maxBlocks;
    unsigned clIdx;
};
