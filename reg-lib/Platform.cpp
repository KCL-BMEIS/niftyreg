#include "Platform.h"
#include "AladinContent.h"
#include "KernelFactory.h"
#include "CpuKernelFactory.h"
#ifdef _USE_CUDA
#include "CudaKernelFactory.h"
#include "CudaContextSingleton.h"
#endif
#ifdef _USE_OPENCL
#include "ClKernelFactory.h"
#include "ClContextSingleton.h"
#endif

using namespace std;

/* *************************************************************** */
Platform::Platform(int platformCode) {
    this->platformCode = platformCode;
    if (platformCode == NR_PLATFORM_CPU) {
        this->factory = new CpuKernelFactory();
        this->platformName = "cpu_platform";
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        this->factory = new CudaKernelFactory();
        this->platformName = "cuda_platform";
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        this->factory = new ClKernelFactory();
        this->platformName = "cl_platform";
    }
#endif
}
/* *************************************************************** */
Kernel* Platform::CreateKernel(const string& name, Content *con) const {
    return this->factory->ProduceKernel(name, con);
}
/* *************************************************************** */
std::string Platform::GetName() {
    return platformName;
}
/* *************************************************************** */
unsigned Platform::GetGpuIdx() {
    return gpuIdx;
}
/* *************************************************************** */
void Platform::SetGpuIdx(unsigned gpuIdxIn) {
    if (platformCode == NR_PLATFORM_CPU) {
        gpuIdx = 999;
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        CudaContextSingleton *cudaContext = &CudaContextSingleton::Instance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            cudaContext->SetCudaIdx(gpuIdxIn);
        }
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        ClContextSingleton *sContext = &ClContextSingleton::Instance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            sContext->SetClIdx(gpuIdxIn);
        }

        std::size_t paramValueSize;
        sContext->checkErrNum(clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_TYPE, 0, nullptr, &paramValueSize), "Failed to find OpenCL device info ");
        cl_device_type *field = (cl_device_type *)alloca(sizeof(cl_device_type) * paramValueSize);
        sContext->checkErrNum(clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_TYPE, paramValueSize, field, nullptr), "Failed to find OpenCL device info ");
        if (CL_DEVICE_TYPE_CPU == *field) {
            reg_print_fct_error("Platform::setClIdx");
            reg_print_msg_error("The OpenCL kernels only support GPU devices for now. Exit");
            reg_exit();
        }
    }
#endif
}
/* *************************************************************** */
int Platform::GetPlatformCode() {
    return platformCode;
}
/* *************************************************************** */
//void Platform::SetPlatformCode(const int platformCodeIn) {
//    this->platformCode = platformCodeIn;
//}
/* *************************************************************** */
Platform::~Platform() {
    delete this->factory;
}
/* *************************************************************** */
