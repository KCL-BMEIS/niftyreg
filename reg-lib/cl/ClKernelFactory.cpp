#include "ClKernelFactory.h"
#include "ClAffineDeformationFieldKernel.h"
#include "ClConvolutionKernel.h"
#include "ClBlockMatchingKernel.h"
#include "ClResampleImageKernel.h"
#include "ClOptimiseKernel.h"
#include "AladinContent.h"

Kernel* ClKernelFactory::ProduceKernel(std::string name, Content *con) const {
	if (name == AffineDeformationFieldKernel::GetName()) return new ClAffineDeformationFieldKernel(con);
	else if (name == ConvolutionKernel::GetName()) return new ClConvolutionKernel();
	else if (name == BlockMatchingKernel::GetName()) return new ClBlockMatchingKernel(con);
	else if (name == ResampleImageKernel::GetName()) return new ClResampleImageKernel(con);
	else if (name == OptimiseKernel::GetName()) return new ClOptimiseKernel(con);
	else return nullptr;
}
