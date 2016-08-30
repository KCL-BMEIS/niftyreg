#include "CPUKernelFactory.h"
#include "CPUAffineDeformationFieldKernel.h"
#include "CPUConvolutionKernel.h"
#include "CPUBlockMatchingKernel.h"
#include "CPUResampleImageKernel.h"
#include "CPUOptimiseKernel.h"
//New kernels
#include "CPUSplineDeformationFieldKernel.h"
#include "CPURefineControlPointGridKernel.h"
#include "CPUDeformationFieldFromVelocityGridKernel.h"

Kernel *CPUKernelFactory::produceKernel(std::string name,  GlobalContent *con) const
{
	if (name == AffineDeformationFieldKernel::getName()) return new CPUAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::getName()) return new CPUConvolutionKernel(name);
	else if (name == BlockMatchingKernel::getName()) return new CPUBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::getName()) return new CPUResampleImageKernel(con, name);
	else if (name == OptimiseKernel::getName()) return new CPUOptimiseKernel(con, name);
    //NEW kernels
    else if (name == SplineDeformationFieldKernel::getName()) return new CPUSplineDeformationFieldKernel(con, name);
    else if (name == RefineControlPointGridKernel::getName()) return new CPURefineControlPointGridKernel(con, name);
    else if (name == DeformationFieldFromVelocityGridKernel::getName()) return new CPUDeformationFieldFromVelocityGridKernel(con, name);
	else return NULL;
}
