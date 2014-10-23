#include "CPUKernelFactory.h"
#include "KernelImpl.h"
#include "CPUKernels.h"
#include "Platform.h"
#include "CpuContext.h"

KernelImpl* CPUKernelFactory::createKernelImpl(std::string name, const Platform& platform, Context* con) const {
	std::cout << "CPU Factory: Creating: " << name << std::endl;
	if (name == AffineDeformationFieldKernel::Name()) return new CPUAffineDeformationFieldKernel(con, name, platform);
	else if (name == ConvolutionKernel::Name()) return new CPUConvolutionKernel(name, platform);
	else if (name == BlockMatchingKernel::Name()) return new CPUBlockMatchingKernel(con, name, platform);
	else if (name == ResampleImageKernel::Name()) return new CPUResampleImageKernel(con, name, platform);
	else if (name == OptimiseKernel::Name()) return new CPUOptimiseKernel(con, name, platform);
	else return NULL;


	//put this on the calling function
	/**/
}

