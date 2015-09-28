#include "CUDAKernelFactory.h"
#include "CUDAAffineDeformationFieldKernel.h"
#include "CUDAConvolutionKernel.h"
#include "CUDABlockMatchingKernel.h"
#include "CUDAResampleImageKernel.h"
#include "CUDAOptimiseKernel.h"
#include "Content.h"

Kernel *CudaKernelFactory::produceKernel(std::string name,  Content *con) const {
	if( name == AffineDeformationFieldKernel::getName() ) return new CudaAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::getName() ) return new CudaConvolutionKernel(name);
	else if( name == BlockMatchingKernel::getName() ) return new CudaBlockMatchingKernel( con, name);
	else if( name == ResampleImageKernel::getName() ) return new CudaResampleImageKernel(con, name);
	else if( name == OptimiseKernel::getName() ) return new CudaOptimiseKernel(con, name);
	else return NULL;
}
