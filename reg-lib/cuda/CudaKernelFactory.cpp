#include "CudaKernelFactory.h"
#include "CudaAffineDeformationFieldKernel.h"
#include "CudaConvolutionKernel.h"
#include "CudaBlockMatchingKernel.h"
#include "CudaResampleImageKernel.h"
#include "CudaLtsKernel.h"
#include "AladinContent.h"

Kernel* CudaKernelFactory::Produce(std::string name, Content *con) const {
    if (name == AffineDeformationFieldKernel::GetName()) return new CudaAffineDeformationFieldKernel(con);
    else if (name == ConvolutionKernel::GetName()) return new CudaConvolutionKernel();
    else if (name == BlockMatchingKernel::GetName()) return new CudaBlockMatchingKernel(con);
    else if (name == ResampleImageKernel::GetName()) return new CudaResampleImageKernel(con);
    else if (name == LtsKernel::GetName()) return new CudaLtsKernel(con);
    else return nullptr;
}
