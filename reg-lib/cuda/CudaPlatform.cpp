#include "CudaPlatform.h"
#include "CudaKernelFactory.h"
#include "Kernels.h"
CudaPlatform::CudaPlatform() {
	//register the necessary kernels for the platform 
	CudaKernelFactory* factory = new CudaKernelFactory();
	assignKernelToFactory(AffineDeformationFieldKernel::getName(), factory);
	assignKernelToFactory(BlockMatchingKernel::getName(), factory);
	assignKernelToFactory(ConvolutionKernel::getName(), factory);
	assignKernelToFactory(OptimiseKernel::getName(), factory);
	assignKernelToFactory(ResampleImageKernel::getName(), factory);
}
