#include "CUDAKernelFactory.h"
#include "CUDAAffineDeformationFieldKernel.h"
#include "CUDAConvolutionKernel.h"
#include "CUDABlockMatchingKernel.h"
#include "CUDAResampleImageKernel.h"
#include "CUDAOptimiseKernel.h"
//New
#include "CUDASplineDeformationFieldKernel.h"
#include "CUDARefineControlPointGridKernel.h"
#include "CUDADeformationFieldFromVelocityGridKernel.h"

Kernel *CUDAKernelFactory::produceKernel(std::string name,  GlobalContent *con) const {
    if( name == AffineDeformationFieldKernel::getName() ) return new CUDAAffineDeformationFieldKernel(con, name);
    else if( name == ConvolutionKernel::getName() ) return new CUDAConvolutionKernel(name);
    else if( name == BlockMatchingKernel::getName() ) return new CUDABlockMatchingKernel( con, name);
    else if( name == ResampleImageKernel::getName() ) return new CUDAResampleImageKernel(con, name);
    else if( name == OptimiseKernel::getName() ) return new CUDAOptimiseKernel(con, name);
    //NEW Kernels
    else if (name == SplineDeformationFieldKernel::getName()) return new CUDASplineDeformationFieldKernel(con, name);
    else if (name == RefineControlPointGridKernel::getName()) return new CUDARefineControlPointGridKernel(con, name);
    else if (name == DeformationFieldFromVelocityGridKernel::getName()) return new CUDADeformationFieldFromVelocityGridKernel(con, name);
	else return NULL;
}
