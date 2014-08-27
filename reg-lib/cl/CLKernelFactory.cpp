#include "CLKernelFactory.h"
#include "KernelImpl.h"
#include "CLKernels.h"
#include "Platform.h"

KernelImpl* CLKernelFactory::createKernelImpl(std::string name, const Platform& platform, unsigned int dType) const {
	std::cout << "CL Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CLAffineDeformationFieldKernel(name, platform);
	else if( name == ConvolutionKernel::Name() ) return new CLConvolutionKernel(name, platform);
	else if( name == BlockMatchingKernel::Name() ) return new CLBlockMatchingKernel(name, platform);
	else if( name == ResampleImageKernel::Name() ) return new CLResampleImageKernel(name, platform);
	else if( name == OptimiseKernel::Name() ) return new CLOptimiseKernel(name, platform);
	else return NULL;

}