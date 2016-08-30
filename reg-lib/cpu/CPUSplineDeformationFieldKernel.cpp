#include "CPUSplineDeformationFieldKernel.h"

CPUSplineDeformationFieldKernel::CPUSplineDeformationFieldKernel(GlobalContent *conIn, std::string nameIn) : SplineDeformationFieldKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<F3DContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->mask = con->getCurrentReferenceMask();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void CPUSplineDeformationFieldKernel::calculate(bool compose) {
    reg_spline_getDeformationField(this->controlPointImage,
                                   this->deformationFieldImage,
                                   this->mask,
                                   compose, //composition
                                   true // bspline
                                   );
}
