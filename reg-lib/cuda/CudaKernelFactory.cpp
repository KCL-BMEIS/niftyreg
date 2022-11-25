#include "CudaKernelFactory.h"
#include "CudaAffineDeformationFieldKernel.h"
#include "CudaConvolutionKernel.h"
#include "CudaBlockMatchingKernel.h"
#include "CudaResampleImageKernel.h"
#include "CudaOptimiseKernel.h"
#include "AladinContent.h"

Kernel* CudaKernelFactory::ProduceKernel(std::string name, Content *con) const {
    if (name == AffineDeformationFieldKernel::GetName()) return new CudaAffineDeformationFieldKernel(con);
    else if (name == ConvolutionKernel::GetName()) return new CudaConvolutionKernel();
    else if (name == BlockMatchingKernel::GetName()) return new CudaBlockMatchingKernel(con);
    else if (name == ResampleImageKernel::GetName()) return new CudaResampleImageKernel(con);
    else if (name == OptimiseKernel::GetName()) return new CudaOptimiseKernel(con);
    else return nullptr;
}
