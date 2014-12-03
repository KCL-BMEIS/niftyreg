#include "CLKernelFactory.h"
#include "CLKernels.h"
#include "Context.h"

Kernel* CLKernelFactory::createKernel(std::string name, Context* con) const {

	if( name == AffineDeformationFieldKernel::getName() ) return new CLAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::getName() ) return new CLConvolutionKernel(name);
	else if (name == BlockMatchingKernel::getName()) return new CLBlockMatchingKernel(con, name);
	else if( name == ResampleImageKernel::getName() ) return new CLResampleImageKernel(con, name);
	else if( name == OptimiseKernel::getName() ) return new CLOptimiseKernel(con, name);
	else return NULL;

}
