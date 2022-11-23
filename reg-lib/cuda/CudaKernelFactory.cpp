#include "CudaKernelFactory.h"
#include "CudaAffineDeformationFieldKernel.h"
#include "CudaConvolutionKernel.h"
#include "CudaBlockMatchingKernel.h"
#include "CudaResampleImageKernel.h"
#include "CudaOptimiseKernel.h"
#include "AladinContent.h"

Kernel* CudaKernelFactory::ProduceKernel(std::string name, AladinContent *con) const {
    if (name == AffineDeformationFieldKernel::GetName()) return new CudaAffineDeformationFieldKernel(con, name);
    else if (name == ConvolutionKernel::GetName()) return new CudaConvolutionKernel(name);
    else if (name == BlockMatchingKernel::GetName()) return new CudaBlockMatchingKernel(con, name);
    else if (name == ResampleImageKernel::GetName()) return new CudaResampleImageKernel(con, name);
    else if (name == OptimiseKernel::GetName()) return new CudaOptimiseKernel(con, name);
    else return nullptr;
}
