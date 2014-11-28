#include "CudaPlatform.h"
#include "CudaKernelFactory.h"
#include "Kernels.h"
CudaPlatform::CudaPlatform() {
	//register the necessary kernels for the platform 
	CudaKernelFactory* factory = new CudaKernelFactory();
	registerKernelFactory(AffineDeformationFieldKernel::Name(), factory);
	registerKernelFactory(BlockMatchingKernel::Name(), factory);
	registerKernelFactory(ConvolutionKernel::Name(), factory);
	registerKernelFactory(OptimiseKernel::Name(), factory);
	registerKernelFactory(ResampleImageKernel::Name(), factory);
}
