#include "CLDeformationFieldFromVelocityGridKernel.h"

ClDeformationFieldFromVelocityGridKernel::ClDeformationFieldFromVelocityGridKernel(GlobalContent *conIn, std::string nameIn) : DeformationFieldFromVelocityGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<ClF3DContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void ClDeformationFieldFromVelocityGridKernel::calculate(bool updateStepNumber) {
    reg_spline_getDefFieldFromVelocityGrid(this->controlPointImage,
                                           this->deformationFieldImage,
                                           updateStepNumber
                                           );
    //4 the moment - to update the gpu
    this->con->setCurrentDeformationField(this->deformationFieldImage);
}
