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
#include "Kernels.h"

using namespace std;

Platform::Platform(int platformCode) {

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

Platform::~Platform()
{
}

Kernel* Platform::createKernel(const string& name, Content* con) const {
	return this->factory->produceKernel(name, con);
}



std::string Platform::getName(){
	return this->platformName;
}

void Platform::setClIdx(int clIdxIn){
#ifdef _USE_OPENCL
	CLContextSingletton *sContext = &CLContextSingletton::Instance();
	sContext->setClIdx(clIdxIn);
#endif
}
