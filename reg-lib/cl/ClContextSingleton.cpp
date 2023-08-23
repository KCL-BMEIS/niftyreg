#include "ClContextSingleton.h"

/* *************************************************************** */
ClContextSingleton::ClContextSingleton() {
    this->commandQueue = nullptr;
    this->context = nullptr;
    this->clIdx = 999;
    Init();
}
/* *************************************************************** */
void ClContextSingleton::Init() {
    // Query the number of platforms
    cl_int errNum = clGetPlatformIDs(0, nullptr, &this->numPlatforms);
    CheckErrNum(errNum, "Failed to find CL platforms.");

    this->platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * this->numPlatforms);
    errNum = clGetPlatformIDs(this->numPlatforms, this->platformIds, nullptr);
    CheckErrNum(errNum, "Failed to find any OpenCL platforms.");

    errNum = clGetDeviceIDs(this->platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &this->numDevices);
    CheckErrNum(errNum, "Failed to find OpenCL devices.");

    this->devices = new cl_device_id[this->numDevices];
    errNum = clGetDeviceIDs(this->platformIds[0], CL_DEVICE_TYPE_ALL, this->numDevices, this->devices, nullptr);

    PickCard(this->clIdx);

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)this->platformIds[0], 0 };
    this->context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &errNum);

    if (errNum != CL_SUCCESS) {
        NR_WARN("Could not create GPU context, trying CPU...");
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, nullptr, nullptr, &errNum);
        if (errNum != CL_SUCCESS) {
            NR_ERROR("Failed to create an OpenCL GPU or CPU context");
            return;
        }
    }

    this->commandQueue = clCreateCommandQueue(this->context, this->devices[this->clIdx], CL_QUEUE_PROFILING_ENABLE, nullptr);
    CheckErrNum(errNum, "Failed to create commandQueue for device ");

    this->deviceId = this->devices[this->clIdx];
    QueryGridDims();
}
/* *************************************************************** */
void ClContextSingleton::SetClIdx(int clIdxIn) {
    clIdx = clIdxIn;
    this->Init();
}
/* *************************************************************** */
void ClContextSingleton::QueryGridDims() {
    std::size_t paramValueSize;
    cl_int errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &paramValueSize);
    CheckErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE");

    size_t *info = (size_t*)alloca(sizeof(size_t) * paramValueSize);
    errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, nullptr);
    CheckErrNum(errNum, "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_GROUP_SIZE2");
    this->maxThreads = *info;
    this->maxBlocks = 65535;
}
/* *************************************************************** */
void ClContextSingleton::PickCard(cl_uint deviceId) {
    cl_int errNum;
    std::size_t paramValueSize;
    cl_uint maxProcs = 0;
    this->clIdx = 0;
    this->isCardDoubleCapable = 0;

    std::size_t paramValueSizeDOUBE1;
    std::size_t paramValueSizeDOUBE2;

    if (deviceId < this->numDevices) {
        this->clIdx = deviceId;
        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &paramValueSize);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint *info = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSize);
        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_MAX_COMPUTE_UNITS, paramValueSize, info, nullptr);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint numProcs = *info;
        maxProcs = numProcs;

        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 0, nullptr, &paramValueSizeDOUBE1);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint *infoD1 = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSizeDOUBE1);
        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, paramValueSizeDOUBE1, infoD1, nullptr);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint numD1 = *infoD1;

        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, 0, nullptr, &paramValueSizeDOUBE2);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint *infoD2 = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSizeDOUBE2);
        errNum = clGetDeviceInfo(this->devices[this->clIdx], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, paramValueSizeDOUBE2, infoD2, nullptr);
        CheckErrNum(errNum, "Failed to find OpenCL device info ");
        cl_uint numD2 = *infoD2;

        if (numD1 > 0 || numD2 > 0) {
            this->isCardDoubleCapable = true;
        } else {
            this->isCardDoubleCapable = false;
        }
        return;
    } else if (deviceId != 999)
        NR_FATAL_ERROR("The specified OpenCL card ID is not defined! Run reg_gpuinfo to get the proper ID.");

    for (cl_uint i = 0; i < this->numDevices; ++i) {
        cl_device_type dev_type;
        clGetDeviceInfo(this->devices[i], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr);
        if (dev_type == CL_DEVICE_TYPE_GPU) {
            errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &paramValueSize);
            CheckErrNum(errNum, "Failed to find OpenCL device info ");
            cl_uint *info = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSize);
            errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, paramValueSize, info, nullptr);
            CheckErrNum(errNum, "Failed to find OpenCL device info ");
            cl_uint numProcs = *info;
            const bool found = numProcs > maxProcs;
            this->clIdx = found ? i : this->clIdx;
            maxProcs = found ? numProcs : maxProcs;

            if (found) {
                errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 0, nullptr, &paramValueSizeDOUBE1);
                CheckErrNum(errNum, "Failed to find OpenCL device info ");
                cl_uint *infoD1 = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSizeDOUBE1);
                errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, paramValueSizeDOUBE1, infoD1, nullptr);
                CheckErrNum(errNum, "Failed to find OpenCL device info ");
                cl_uint numD1 = *infoD1;

                errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, 0, nullptr, &paramValueSizeDOUBE2);
                CheckErrNum(errNum, "Failed to find OpenCL device info ");
                cl_uint *infoD2 = (cl_uint*)alloca(sizeof(cl_uint) * paramValueSizeDOUBE2);
                errNum = clGetDeviceInfo(this->devices[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, paramValueSizeDOUBE2, infoD2, nullptr);
                CheckErrNum(errNum, "Failed to find OpenCL device info ");
                cl_uint numD2 = *infoD2;

                if (numD1 > 0 || numD2 > 0) {
                    this->isCardDoubleCapable = true;
                } else {
                    this->isCardDoubleCapable = false;
                }
            }
        }
    }
}
/* *************************************************************** */
cl_program ClContextSingleton::CreateProgram(const char *fileName) {
    cl_int errNum;
    cl_program program;
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        NR_ERROR("Failed to open file for reading: " << fileName);
        return nullptr;
    }
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(this->context, 1, (const char**)&srcStr, nullptr, &errNum);
    CheckErrNum(errNum, "Failed to create CL program");

    errNum = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (errNum != CL_SUCCESS) {
        CheckDebugKernelInfo(program, this->deviceId, "Errors in kernel: ");
        //create log
        char buffer[2048];
        clGetProgramBuildInfo(program, this->devices[this->clIdx], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, nullptr);
        NR_FATAL_ERROR("--- Build log ---\n"s + buffer);
    }

    return program;
}
/* *************************************************************** */
ClContextSingleton::~ClContextSingleton() {
    if (this->context != 0) clReleaseContext(this->context);
    if (this->commandQueue != 0) clReleaseCommandQueue(this->commandQueue);
    delete[] this->devices;
}
/* *************************************************************** */
void ClContextSingleton::CheckDebugKernelInfo(cl_program program, cl_device_id devIdIn, const char *message) {
    char buffer[10240];
    clGetProgramBuildInfo(program, devIdIn, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, nullptr);
    NR_ERROR(message);
    NR_ERROR(buffer);
}
/* *************************************************************** */
void ClContextSingleton::CheckErrNum(cl_int errNum, std::string message) {
    if (errNum != CL_SUCCESS) {
        NR_ERROR(message);
        switch (errNum) {
        case -1: NR_FATAL_ERROR("CL_DEVICE_NOT_FOUND");
        case -2: NR_FATAL_ERROR("CL_DEVICE_NOT_AVAILABLE");
        case -3: NR_FATAL_ERROR("CL_COMPILER_NOT_AVAILABLE");
        case -4: NR_FATAL_ERROR("CL_MEM_OBJECT_ALLOCATION_FAILURE");
        case -5: NR_FATAL_ERROR("CL_OUT_OF_RESOURCES");
        case -6: NR_FATAL_ERROR("CL_OUT_OF_HOST_MEMORY");
        case -7: NR_FATAL_ERROR("CL_PROFILING_INFO_NOT_AVAILABLE");
        case -8: NR_FATAL_ERROR("CL_MEM_COPY_OVERLAP");
        case -9: NR_FATAL_ERROR("CL_IMAGE_FORMAT_MISMATCH");
        case -10: NR_FATAL_ERROR("CL_IMAGE_FORMAT_NOT_SUPPORTED");
        case -11: NR_FATAL_ERROR("CL_BUILD_PROGRAM_FAILURE");
        case -12: NR_FATAL_ERROR("CL_MAP_FAILURE");
        case -13: NR_FATAL_ERROR("CL_MISALIGNED_SUB_BUFFER_OFFSET");
        case -14: NR_FATAL_ERROR("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
        case -15: NR_FATAL_ERROR("CL_COMPILE_PROGRAM_FAILURE");
        case -16: NR_FATAL_ERROR("CL_LINKER_NOT_AVAILABLE");
        case -17: NR_FATAL_ERROR("CL_LINK_PROGRAM_FAILURE");
        case -18: NR_FATAL_ERROR("CL_DEVICE_PARTITION_FAILED");
        case -19: NR_FATAL_ERROR("CL_KERNEL_ARG_INFO_NOT_AVAILABLE");
        case -30: NR_FATAL_ERROR("CL_INVALID_VALUE");
        case -31: NR_FATAL_ERROR("CL_INVALID_DEVICE_TYPE");
        case -32: NR_FATAL_ERROR("CL_INVALID_PLATFORM");
        case -33: NR_FATAL_ERROR("CL_INVALID_DEVICE");
        case -34: NR_FATAL_ERROR("CL_INVALID_CONTEXT");
        case -35: NR_FATAL_ERROR("CL_INVALID_QUEUE_PROPERTIES");
        case -36: NR_FATAL_ERROR("CL_INVALID_COMMAND_QUEUE");
        case -37: NR_FATAL_ERROR("CL_INVALID_HOST_PTR");
        case -38: NR_FATAL_ERROR("CL_INVALID_MEM_OBJECT");
        case -39: NR_FATAL_ERROR("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
        case -40: NR_FATAL_ERROR("CL_INVALID_IMAGE_SIZE");
        case -41: NR_FATAL_ERROR("CL_INVALID_SAMPLER");
        case -42: NR_FATAL_ERROR("CL_INVALID_BINARY");
        case -43: NR_FATAL_ERROR("CL_INVALID_BUILD_OPTIONS");
        case -44: NR_FATAL_ERROR("CL_INVALID_PROGRAM");
        case -45: NR_FATAL_ERROR("CL_INVALID_PROGRAM_EXECUTABLE");
        case -46: NR_FATAL_ERROR("CL_INVALID_KERNEL_NAME");
        case -47: NR_FATAL_ERROR("CL_INVALID_KERNEL_DEFINITION");
        case -48: NR_FATAL_ERROR("CL_INVALID_KERNEL");
        case -49: NR_FATAL_ERROR("CL_INVALID_ARG_INDEX");
        case -50: NR_FATAL_ERROR("CL_INVALID_ARG_VALUE");
        case -51: NR_FATAL_ERROR("CL_INVALID_ARG_SIZE");
        case -52: NR_FATAL_ERROR("CL_INVALID_KERNEL_ARGS");
        case -53: NR_FATAL_ERROR("CL_INVALID_WORK_DIMENSION");
        case -54: NR_FATAL_ERROR("CL_INVALID_WORK_GROUP_SIZE");
        case -55: NR_FATAL_ERROR("CL_INVALID_WORK_ITEM_SIZE");
        case -56: NR_FATAL_ERROR("CL_INVALID_GLOBAL_OFFSET");
        case -57: NR_FATAL_ERROR("CL_INVALID_EVENT_WAIT_LIST");
        case -58: NR_FATAL_ERROR("CL_INVALID_EVENT");
        case -59: NR_FATAL_ERROR("CL_INVALID_OPERATION");
        case -60: NR_FATAL_ERROR("CL_INVALID_GL_OBJECT");
        case -61: NR_FATAL_ERROR("CL_INVALID_BUFFER_SIZE");
        case -62: NR_FATAL_ERROR("CL_INVALID_MIP_LEVEL");
        case -63: NR_FATAL_ERROR("CL_INVALID_GLOBAL_WORK_SIZE");
        case -64: NR_FATAL_ERROR("CL_INVALID_PROPERTY");
        case -65: NR_FATAL_ERROR("CL_INVALID_IMAGE_DESCRIPTOR");
        case -66: NR_FATAL_ERROR("CL_INVALID_COMPILER_OPTIONS");
        case -67: NR_FATAL_ERROR("CL_INVALID_LINKER_OPTIONS");
        case -68: NR_FATAL_ERROR("CL_INVALID_DEVICE_PARTITION_COUNT");
        default: NR_FATAL_ERROR("Unknown error type");
        }
    }
}
/* *************************************************************** */
cl_context ClContextSingleton::GetContext() {
    return this->context;
}
/* *************************************************************** */
cl_device_id ClContextSingleton::GetDeviceId() {
    return this->deviceId;
}
/* *************************************************************** */
cl_device_id* ClContextSingleton::GetDevices() {
    return this->devices;
}
/* *************************************************************** */
cl_command_queue ClContextSingleton::GetCommandQueue() {
    return this->commandQueue;
}
/* *************************************************************** */
cl_uint ClContextSingleton::GetNumPlatforms() {
    return this->numPlatforms;
}
/* *************************************************************** */
cl_platform_id* ClContextSingleton::GetPlatformIds() {
    return this->platformIds;
}
/* *************************************************************** */
cl_uint ClContextSingleton::GetNumDevices() {
    return this->numDevices;
}
/* *************************************************************** */
size_t ClContextSingleton::GetMaxThreads() {
    return this->maxThreads;
}
/* *************************************************************** */
bool ClContextSingleton::IsCardDoubleCapable() {
    return this->isCardDoubleCapable;
}
/* *************************************************************** */
unsigned ClContextSingleton::GetMaxBlocks() {
    return this->maxBlocks;
}
/* *************************************************************** */
size_t ClContextSingleton::GetWarpGroupLength(cl_kernel kernel) {
    size_t local;
    // Get the maximum work group size for executing the kernel on the device
    cl_int err = clGetKernelWorkGroupInfo(kernel, this->deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, nullptr);
    CheckErrNum(err, "Error: Failed to retrieve kernel work group info!");
    return local;
}
/* *************************************************************** */
cl_kernel ClContextSingleton::DummyKernel(cl_device_id deviceIdIn) {
    const char *source = "\n"
        "__kernel void dummy(                                \n"
        "   __global float* in,                              \n"
        "   __global float* out,                             \n"
        "   const unsigned count)                            \n"
        "{                                                   \n"
        "   int i = get_global_id(0);                        \n"
        "   if(i < count)                                    \n"
        "       out[i] = in[i] * out[i];                     \n"
        "}                                                   \n"
        "\n";

    cl_int  err;
    cl_program program = clCreateProgramWithSource(this->context, 1, (const char **)&source, nullptr, &err);
    CheckErrNum(err, "Failed to create CL program");
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) CheckDebugKernelInfo(program, deviceIdIn, "Errors in kernel: ");

    // Create the compute kernel in the program we wish to run
    cl_kernel kernel = clCreateKernel(program, "dummy", &err);
    if (!kernel || err != CL_SUCCESS) {
        NR_ERROR("Failed to create the compute kernel!");
        return nullptr;
    }
    return kernel;
}
/* *************************************************************** */
