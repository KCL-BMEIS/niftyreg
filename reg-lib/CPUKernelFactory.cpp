#include "CPUKernelFactory.h"
#include "CPUKernels.h"
#include "Context.h"

Kernel* CPUKernelFactory::createKernel(std::string name,  Context* con) const {
	if (name == AffineDeformationFieldKernel::Name()) return new CPUAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::Name()) return new CPUConvolutionKernel(name);
	else if (name == BlockMatchingKernel::Name()) return new CPUBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::Name()) return new CPUResampleImageKernel(con, name);
	else if (name == OptimiseKernel::Name()) return new CPUOptimiseKernel(con, name);
	else return NULL;


	//put this on the calling function
	/**/
}

