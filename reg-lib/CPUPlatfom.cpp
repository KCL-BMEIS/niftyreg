#include "CPUPlatform.h"
#include "CPUKernelFactory.h"
#include "kernels.h"
CPUPlatform::CPUPlatform() {

	//register the necessary kernels for the platform 
	CPUKernelFactory* factory = new CPUKernelFactory();
	registerKernelFactory(AffineDeformationFieldKernel::Name(), factory);
	registerKernelFactory(BlockMatchingKernel::Name(), factory);
	registerKernelFactory(ConvolutionKernel::Name(), factory);
	registerKernelFactory(OptimiseKernel::Name(), factory);
	registerKernelFactory(ResampleImageKernel::Name(), factory);
	

}
