#include "CPUKernelFactory.h"
#include "CPUKernels.h"
#include "Content.h"

Kernel *CPUKernelFactory::produceKernel(std::string name,  Content *con) const
{
	if (name == AffineDeformationFieldKernel::getName()) return new CPUAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::getName()) return new CPUConvolutionKernel(name);
	else if (name == BlockMatchingKernel::getName()) return new CPUBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::getName()) return new CPUResampleImageKernel(con, name);
	else if (name == OptimiseKernel::getName()) return new CPUOptimiseKernel(con, name);
	else return NULL;
}

