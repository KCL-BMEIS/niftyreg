#include "ClRefineControlPointGridKernel.h"

ClRefineControlPointGridKernel::ClRefineControlPointGridKernel(GlobalContent *conIn, std::string nameIn) : RefineControlPointGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<ClF3DContent*>(conIn);
    this->referenceImage = con->getCurrentReference();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void ClRefineControlPointGridKernel::calculate() {
    reg_spline_refineControlPointGrid(this->controlPointImage,
                                      this->referenceImage);
    //4 the moment - to update the gpu
    this->con->setCurrentControlPointGrid(this->controlPointImage);
}
