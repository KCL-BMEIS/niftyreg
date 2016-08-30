#include "CLKernelFactory.h"
#include "CLAffineDeformationFieldKernel.h"
#include "CLConvolutionKernel.h"
#include "CLBlockMatchingKernel.h"
#include "CLResampleImageKernel.h"
#include "CLOptimiseKernel.h"
//New
#include "CLSplineDeformationFieldKernel.h"
#include "CLRefineControlPointGridKernel.h"
#include "CLDeformationFieldFromVelocityGridKernel.h"

Kernel *CLKernelFactory::produceKernel(std::string name, GlobalContent *con) const {

	if( name == AffineDeformationFieldKernel::getName() ) return new CLAffineDeformationFieldKernel(con, name);
	else if( name == ConvolutionKernel::getName() ) return new CLConvolutionKernel(name);
	else if (name == BlockMatchingKernel::getName()) return new CLBlockMatchingKernel(con, name);
	else if( name == ResampleImageKernel::getName() ) return new CLResampleImageKernel(con, name);
	else if( name == OptimiseKernel::getName() ) return new CLOptimiseKernel(con, name);
    //NEW Kernels
    else if (name == SplineDeformationFieldKernel::getName()) return new ClSplineDeformationFieldKernel(con, name);
    else if (name == RefineControlPointGridKernel::getName()) return new ClRefineControlPointGridKernel(con, name);
    else if (name == DeformationFieldFromVelocityGridKernel::getName()) return new ClDeformationFieldFromVelocityGridKernel(con, name);
	else return NULL;
}
