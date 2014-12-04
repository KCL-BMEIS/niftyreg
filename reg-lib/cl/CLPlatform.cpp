#include "CLPlatform.h"
#include "CLKernelFactory.h"
#include "Kernels.h"
CLPlatform::CLPlatform() {

	//register the necessary kernels for the platform 
	CLKernelFactory* factory = new CLKernelFactory();
	assignKernelToFactory(AffineDeformationFieldKernel::getName(), factory);
	assignKernelToFactory(BlockMatchingKernel::getName(), factory);
	assignKernelToFactory(ConvolutionKernel::getName(), factory);
	assignKernelToFactory(OptimiseKernel::getName(), factory);
	assignKernelToFactory(ResampleImageKernel::getName(), factory);
}
