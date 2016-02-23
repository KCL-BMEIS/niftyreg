#include "Platform.h"
#include "AladinContent.h"
#include "KernelFactory.h"
#include "CPUKernelFactory.h"
#ifdef _USE_CUDA
#include "CUDAKernelFactory.h"
#include "CUDAContextSingletton.h"
#endif
#ifdef _USE_OPENCL
#include "CLKernelFactory.h"
#include "CLContextSingletton.h"
#endif

using namespace std;

/* *************************************************************** */
Platform::Platform(int platformCode)
{
    this->platformCode = platformCode;
    if (platformCode == NR_PLATFORM_CPU) {
        this->factory = new CPUKernelFactory();
        this->platformName = "cpu_platform";
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        this->factory = new CUDAKernelFactory();
        this->platformName = "cuda_platform";
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        this->factory = new CLKernelFactory();
        this->platformName = "cl_platform";
    }
#endif
}
/* *************************************************************** */
Kernel *Platform::createKernel(const string& name, AladinContent *con) const
{
    return this->factory->produceKernel(name, con);
}
/* *************************************************************** */
std::string Platform::getName()
{
    return this->platformName;
}
/* *************************************************************** */
unsigned Platform::getGpuIdx()
{
    return this->gpuIdx;
}
/* *************************************************************** */
void Platform::setGpuIdx(unsigned gpuIdxIn)
{
    if(this->platformCode == NR_PLATFORM_CPU)
    {
        this->gpuIdx = 999;
    }
#ifdef _USE_CUDA
    else if(this->platformCode == NR_PLATFORM_CUDA) {
            CUDAContextSingletton *cudaContext = &CUDAContextSingletton::Instance();
            if(gpuIdxIn != 999) {
                this->gpuIdx = gpuIdxIn;
                cudaContext->setCudaIdx(gpuIdxIn);
            }
        }
#endif
#ifdef _USE_OPENCL
    else if(this->platformCode == NR_PLATFORM_CL) {
            CLContextSingletton *sContext = &CLContextSingletton::Instance();
            if(gpuIdxIn != 999) {
                this->gpuIdx = gpuIdxIn;
                sContext->setClIdx(gpuIdxIn);
            }

            std::size_t paramValueSize;
            sContext->checkErrNum(clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_TYPE, 0, NULL, &paramValueSize), "Failed to find OpenCL device info ");
            cl_device_type *field = (cl_device_type *) alloca(sizeof(cl_device_type) * paramValueSize);
            sContext->checkErrNum(clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_TYPE, paramValueSize, field, NULL), "Failed to find OpenCL device info ");
            if(CL_DEVICE_TYPE_CPU==*field){
                reg_print_fct_error("Platform::setClIdx");
                reg_print_msg_error("The OpenCL kernels only support GPU devices for now. Exit");
                reg_exit();
            }
        }
#endif
}
/* *************************************************************** */
int Platform::getPlatformCode() {
    return this->platformCode;
}
/* *************************************************************** */
//void Platform::setPlatformCode(const int platformCodeIn) {
//    this->platformCode = platformCodeIn;
//}
/* *************************************************************** */
Platform::~Platform()
{
    delete this->factory;
}
/* *************************************************************** */
