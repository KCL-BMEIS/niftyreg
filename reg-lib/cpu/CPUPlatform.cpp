#include "CPUPlatform.h"
#include "CPUKernelFactory.h"
#include "Kernels.h"
CPUPlatform::CPUPlatform() {

	//register the necessary kernels for the platform 
	CPUKernelFactory* factory = new CPUKernelFactory();
	assignKernelToFactory(AffineDeformationFieldKernel::getName(), factory);
	assignKernelToFactory(BlockMatchingKernel::getName(), factory);
	assignKernelToFactory(ConvolutionKernel::getName(), factory);
	assignKernelToFactory(OptimiseKernel::getName(), factory);
	assignKernelToFactory(ResampleImageKernel::getName(), factory);
	

}
