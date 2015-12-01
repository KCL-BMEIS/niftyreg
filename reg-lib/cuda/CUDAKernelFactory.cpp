#include "CUDAKernelFactory.h"
#include "CUDAAffineDeformationFieldKernel.h"
#include "CUDAConvolutionKernel.h"
#include "CUDABlockMatchingKernel.h"
#include "CUDAResampleImageKernel.h"
#include "CUDAOptimiseKernel.h"
#include "AladinContent.h"

Kernel *CUDAKernelFactory::produceKernel(std::string name,  AladinContent *con) const {
    if( name == AffineDeformationFieldKernel::getName() ) return new CUDAAffineDeformationFieldKernel(con, name);
    else if( name == ConvolutionKernel::getName() ) return new CUDAConvolutionKernel(name);
    else if( name == BlockMatchingKernel::getName() ) return new CUDABlockMatchingKernel( con, name);
    else if( name == ResampleImageKernel::getName() ) return new CUDAResampleImageKernel(con, name);
    else if( name == OptimiseKernel::getName() ) return new CUDAOptimiseKernel(con, name);
	else return NULL;
}
