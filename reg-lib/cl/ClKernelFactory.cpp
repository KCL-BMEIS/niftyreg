#include "ClKernelFactory.h"
#include "ClAffineDeformationFieldKernel.h"
#include "ClConvolutionKernel.h"
#include "ClBlockMatchingKernel.h"
#include "ClResampleImageKernel.h"
#include "ClOptimiseKernel.h"
#include "AladinContent.h"

Kernel* ClKernelFactory::ProduceKernel(std::string name, AladinContent *con) const {

	if (name == AffineDeformationFieldKernel::GetName()) return new ClAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::GetName()) return new ClConvolutionKernel(name);
	else if (name == BlockMatchingKernel::GetName()) return new ClBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::GetName()) return new ClResampleImageKernel(con, name);
	else if (name == OptimiseKernel::GetName()) return new ClOptimiseKernel(con, name);
	else return nullptr;
}
