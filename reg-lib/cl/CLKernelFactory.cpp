#include "CLKernelFactory.h"
#include "KernelImpl.h"
#include "CLKernels.h"
#include "Platform.h"
#include "CLContext.h"

Kernel* CLKernelFactory::createKernel(std::string name, Context* con) const {
	//std::cout << "CL Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CLAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::Name() ) return new CLConvolutionKernel(name);
	else if (name == BlockMatchingKernel::Name()) return new CLBlockMatchingKernel(con, name);
	else if( name == ResampleImageKernel::Name() ) return new CLResampleImageKernel(con, name);
	else if( name == OptimiseKernel::Name() ) return new CLOptimiseKernel(con, name);
	else return NULL;

}