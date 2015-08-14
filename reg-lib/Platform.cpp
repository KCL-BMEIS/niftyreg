#include "Platform.h"
#include "Content.h"
#include "KernelFactory.h"
#include "CPUKernelFactory.h"
#ifdef _USE_CUDA
#include "CudaKernelFactory.h"
#endif
#ifdef _USE_OPENCL
#include "CLKernelFactory.h"
#include "CLContextSingletton.h"
#endif
//#include "Kernels.h"

using namespace std;

/* *************************************************************** */
Platform::Platform(int platformCode)
{
	if (platformCode == NR_PLATFORM_CPU) {
		this->factory = new CPUKernelFactory();
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
		this->factory = new CLKernelFactory();
		this->platformName = "cl_platform";
	}
#endif
}
/* *************************************************************** */
Kernel *Platform::createKernel(const string& name, Content *con) const
{
	return this->factory->produceKernel(name, con);
}
/* *************************************************************** */
std::string Platform::getName()
{
	return this->platformName;
}
/* *************************************************************** */
void Platform::setClIdx(int clIdxIn)
{
#ifdef _USE_OPENCL
	CLContextSingletton *sContext = &CLContextSingletton::Instance();
	sContext->setClIdx(clIdxIn);
	std::size_t paramValueSize;
	sContext->checkErrNum(clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_TYPE, 0, NULL, &paramValueSize), "Failed to find OpenCL device info ");
	cl_device_type *field = (cl_device_type *) alloca(sizeof(cl_device_type) * paramValueSize);
	sContext->checkErrNum(clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_TYPE, paramValueSize, field, NULL), "Failed to find OpenCL device info ");
	if(CL_DEVICE_TYPE_CPU==*field){
		reg_print_fct_error("Platform::setClIdx");
		reg_print_msg_error("The OpenCL kernels only support GPU devices for now. Exit");
		reg_exit(1);
	}
#endif
}
/* *************************************************************** */
Platform::~Platform(){}
/* *************************************************************** */
