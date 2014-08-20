#include "CudaKernelFactory.h"
#include "KernelImpl.h"
#include "CudaKernels.h"
#include "Platform.h"

KernelImpl* CudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, unsigned int dType) const {
	std::cout << "CUDA Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CudaAffineDeformationFieldKernel(name, platform);
	else if( name == ConvolutionKernel::Name() ) return new CudaConvolutionKernel(name, platform);
	else if( name == BlockMatchingKernel::Name() ) return new CudaBlockMatchingKernel(name, platform);
	else if( name == ResampleImageKernel::Name() ) return new CudaResampleImageKernel(name, platform);
	else if( name == OptimiseKernel::Name() ) return new CudaOptimiseKernel(name, platform);
	else return NULL;

}