#include "CPUKernelFactory.h"
#include "KernelImpl.h"
#include "CPUKernels.h"
#include "Platform.h"

KernelImpl* CPUKernelFactory::createKernelImpl(std::string name, const Platform& platform, unsigned int dType) const {
	std::cout << "CPU Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CPUAffineDeformationFieldKernel(name, platform);
	else if( name == ConvolutionKernel::Name() ) return new CPUConvolutionKernel(name, platform);
	else if( name == BlockMatchingKernel::Name() ) return new CPUBlockMatchingKernel(name, platform);
	else if( name == ResampleImageKernel::Name() ) return new CPUResampleImageKernel(name, platform);
	else if( name == OptimiseKernel::Name() ) return new CPUOptimiseKernel(name, platform);
	else return NULL;


	//put this on the calling function
	/**/
}

