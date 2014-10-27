#include "CPUKernelFactory.h"
#include "KernelImpl.h"
#include "CPUKernels.h"
#include "Platform.h"
#include "CpuContext.h"

Kernel* CPUKernelFactory::createKernel(std::string name,  Context* con) const {
	std::cout << "CPU Factory: Creating: " << name << std::endl;
	if (name == AffineDeformationFieldKernel::Name()) return new CPUAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::Name()) return new CPUConvolutionKernel(name);
	else if (name == BlockMatchingKernel::Name()) return new CPUBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::Name()) return new CPUResampleImageKernel(con, name);
	else if (name == OptimiseKernel::Name()) return new CPUOptimiseKernel(con, name);
	else return NULL;


	//put this on the calling function
	/**/
}

