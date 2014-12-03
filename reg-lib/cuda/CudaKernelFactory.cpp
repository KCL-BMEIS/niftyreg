#include "CudaKernelFactory.h"
#include "CudaKernels.h"
#include "Context.h"

Kernel* CudaKernelFactory::createKernel(std::string name,  Context* con) const {
	if( name == AffineDeformationFieldKernel::getName() ) return new CudaAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::getName() ) return new CudaConvolutionKernel(name);
	else if( name == BlockMatchingKernel::getName() ) return new CudaBlockMatchingKernel( con, name);
	else if( name == ResampleImageKernel::getName() ) return new CudaResampleImageKernel(con, name);
	else if( name == OptimiseKernel::getName() ) return new CudaOptimiseKernel(con, name);
	else return NULL;

}
