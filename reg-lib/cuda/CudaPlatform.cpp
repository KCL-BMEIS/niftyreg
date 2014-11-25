#include "CudaPlatform.h"
#include "CudaKernelFactory.h"
#include "Kernels.h"
CudaPlatform::CudaPlatform() {
	std::cout<<"Cuda Pl 1"<<std::endl;
	//register the necessary kernels for the platform 
	CudaKernelFactory* factory = new CudaKernelFactory();
	registerKernelFactory(AffineDeformationFieldKernel::Name(), factory);
	registerKernelFactory(BlockMatchingKernel::Name(), factory);
	registerKernelFactory(ConvolutionKernel::Name(), factory);
	registerKernelFactory(OptimiseKernel::Name(), factory);
	registerKernelFactory(ResampleImageKernel::Name(), factory);
}
