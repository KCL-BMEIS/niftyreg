#include "ClKernelFactory.h"
#include "ClAffineDeformationFieldKernel.h"
#include "ClConvolutionKernel.h"
#include "ClBlockMatchingKernel.h"
#include "ClResampleImageKernel.h"
#include "ClLtsKernel.h"
#include "AladinContent.h"

Kernel* ClKernelFactory::Produce(std::string name, Content *con) const {
	if (name == AffineDeformationFieldKernel::GetName()) return new ClAffineDeformationFieldKernel(con);
	else if (name == ConvolutionKernel::GetName()) return new ClConvolutionKernel();
	else if (name == BlockMatchingKernel::GetName()) return new ClBlockMatchingKernel(con);
	else if (name == ResampleImageKernel::GetName()) return new ClResampleImageKernel(con);
	else if (name == LtsKernel::GetName()) return new ClLtsKernel(con);
	else return nullptr;
}
