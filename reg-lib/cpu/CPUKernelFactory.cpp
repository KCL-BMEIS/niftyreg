#include "CPUKernelFactory.h"
#include "CPUKernels.h"
#include "Context.h"

Kernel* CPUKernelFactory::createKernel(std::string name,  Context* con) const {
	if (name == AffineDeformationFieldKernel::getName()) return new CPUAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::getName()) return new CPUConvolutionKernel(name);
	else if (name == BlockMatchingKernel::getName()) return new CPUBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::getName()) return new CPUResampleImageKernel(con, name);
	else if (name == OptimiseKernel::getName()) return new CPUOptimiseKernel(con, name);
	else return NULL;


	//put this on the calling function
	/**/
}

