#include "CPUDeformationFieldFromVelocityGridKernel.h"

CPUDeformationFieldFromVelocityGridKernel::CPUDeformationFieldFromVelocityGridKernel(GlobalContent *conIn, std::string nameIn) : DeformationFieldFromVelocityGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<F3DContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void CPUDeformationFieldFromVelocityGridKernel::calculate(bool updateStepNumber) {
    reg_spline_getDefFieldFromVelocityGrid(this->controlPointImage,
                                           this->deformationFieldImage,
                                           updateStepNumber
                                           );
}
