#include "CLSplineDeformationFieldKernel.h"

ClSplineDeformationFieldKernel::ClSplineDeformationFieldKernel(GlobalContent *conIn, std::string nameIn) : SplineDeformationFieldKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<ClF3DContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->mask = con->getCurrentReferenceMask();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void ClSplineDeformationFieldKernel::calculate(bool compose) {
    reg_spline_getDeformationField(this->controlPointImage,
                                   this->deformationFieldImage,
                                   this->mask,
                                   compose, //composition
                                   true // bspline
                                   );
    //4 the moment - to update the gpu
    this->con->setCurrentDeformationField(this->deformationFieldImage);
}
