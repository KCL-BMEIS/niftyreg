#include "CpuKernelFactory.h"
#include "CpuAffineDeformationFieldKernel.h"
#include "CpuConvolutionKernel.h"
#include "CpuBlockMatchingKernel.h"
#include "CpuResampleImageKernel.h"
#include "CpuLtsKernel.h"
#include "AladinContent.h"

Kernel* CpuKernelFactory::Produce(std::string name, Content *con) const {
	if (name == AffineDeformationFieldKernel::GetName()) return new CpuAffineDeformationFieldKernel(con);
	else if (name == ConvolutionKernel::GetName()) return new CpuConvolutionKernel();
	else if (name == BlockMatchingKernel::GetName()) return new CpuBlockMatchingKernel(con);
	else if (name == ResampleImageKernel::GetName()) return new CpuResampleImageKernel(con);
	else if (name == LtsKernel::GetName()) return new CpuLtsKernel(con);
	else return nullptr;
}
