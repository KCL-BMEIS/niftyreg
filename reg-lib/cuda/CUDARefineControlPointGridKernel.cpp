#include "CUDARefineControlPointGridKernel.h"

CUDARefineControlPointGridKernel::CUDARefineControlPointGridKernel(GlobalContent *conIn, std::string nameIn) : RefineControlPointGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<CudaF3DContent*>(conIn);
    this->referenceImage = con->getCurrentReference();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void CUDARefineControlPointGridKernel::calculate() {
    reg_spline_refineControlPointGrid(this->controlPointImage,
                                      this->referenceImage);
    //4 the moment - to update the gpu
    this->con->setCurrentControlPointGrid(this->controlPointImage);
}
