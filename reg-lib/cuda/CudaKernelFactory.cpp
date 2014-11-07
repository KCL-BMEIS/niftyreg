#include "CudaKernelFactory.h"
#include "KernelImpl.h"
#include "CudaKernels.h"
#include "Platform.h"

Kernel* CudaKernelFactory::createKernel(std::string name,  Context* con) const {
	//std::cout << "CUDA Factory called!" << std::endl;
	if( name == AffineDeformationFieldKernel::Name() ) return new CudaAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::Name() ) return new CudaConvolutionKernel(name);
	else if( name == BlockMatchingKernel::Name() ) return new CudaBlockMatchingKernel( con, name);
	else if( name == ResampleImageKernel::Name() ) return new CudaResampleImageKernel(con, name);
	else if( name == OptimiseKernel::Name() ) return new CudaOptimiseKernel(con, name);
	else return NULL;

}