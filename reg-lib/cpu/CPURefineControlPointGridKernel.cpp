#include "CPURefineControlPointGridKernel.h"

CPURefineControlPointGridKernel::CPURefineControlPointGridKernel(GlobalContent *conIn, std::string nameIn) : RefineControlPointGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<F3DContent*>(conIn);
    this->referenceImage = con->getCurrentReference();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void CPURefineControlPointGridKernel::calculate() {
    reg_spline_refineControlPointGrid(this->controlPointImage,
                                      this->referenceImage);
}
