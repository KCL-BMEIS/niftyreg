#include "CpuKernelFactory.h"
#include "CpuAffineDeformationFieldKernel.h"
#include "CpuConvolutionKernel.h"
#include "CpuBlockMatchingKernel.h"
#include "CpuResampleImageKernel.h"
#include "CpuOptimiseKernel.h"
#include "AladinContent.h"

Kernel* CpuKernelFactory::ProduceKernel(std::string name, AladinContent *con) const {
	if (name == AffineDeformationFieldKernel::GetName()) return new CpuAffineDeformationFieldKernel(con, name);
	else if (name == ConvolutionKernel::GetName()) return new CpuConvolutionKernel(name);
	else if (name == BlockMatchingKernel::GetName()) return new CpuBlockMatchingKernel(con, name);
	else if (name == ResampleImageKernel::GetName()) return new CpuResampleImageKernel(con, name);
	else if (name == OptimiseKernel::GetName()) return new CpuOptimiseKernel(con, name);
	else return nullptr;
}
