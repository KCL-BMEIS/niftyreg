#include "CLPlatform.h"
#include "CLKernelFactory.h"
#include "Kernels.h"
CLPlatform::CLPlatform() {

	//register the necessary kernels for the platform 
	CLKernelFactory* factory = new CLKernelFactory();
	registerKernelFactory(AffineDeformationFieldKernel::Name(), factory);
	registerKernelFactory(BlockMatchingKernel::Name(), factory);
	registerKernelFactory(ConvolutionKernel::Name(), factory);
	registerKernelFactory(OptimiseKernel::Name(), factory);
	registerKernelFactory(ResampleImageKernel::Name(), factory);
}
