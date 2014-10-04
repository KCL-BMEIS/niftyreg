#include "CLKernelFactory.h"
#include "KernelImpl.h"
#include "CLKernels.h"
#include "Platform.h"
#include "CLContext.h"

KernelImpl* CLKernelFactory::createKernelImpl(std::string name, const Platform& platform, Context* con) const {
	std::cout << "CL Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CLAffineDeformationFieldKernel(con, name, platform);
	else if( name == ConvolutionKernel::Name() ) return new CLConvolutionKernel(name, platform);
	else if (name == BlockMatchingKernel::Name()) return new CLBlockMatchingKernel(con, name, platform);
	else if( name == ResampleImageKernel::Name() ) return new CLResampleImageKernel(con, name, platform);
	else if( name == OptimiseKernel::Name() ) return new CLOptimiseKernel(con, name, platform);
	else return NULL;

}