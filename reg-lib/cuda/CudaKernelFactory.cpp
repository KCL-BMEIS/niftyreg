#include "CudaKernelFactory.h"
#include "KernelImpl.h"
#include "CudaKernels.h"
#include "Platform.h"

KernelImpl* CudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, Context* con) const {
	std::cout << "CUDA Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CudaAffineDeformationFieldKernel(con, name, platform);
	else if( name == ConvolutionKernel::Name() ) return new CudaConvolutionKernel(name, platform);
	else if( name == BlockMatchingKernel::Name() ) return new CudaBlockMatchingKernel( con, name, platform);
	else if( name == ResampleImageKernel::Name() ) return new CudaResampleImageKernel(con, name, platform);
	else if( name == OptimiseKernel::Name() ) return new CudaOptimiseKernel(con, name, platform);
	else return NULL;

}