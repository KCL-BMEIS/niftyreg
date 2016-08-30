#include "CUDADeformationFieldFromVelocityGridKernel.h"

CUDADeformationFieldFromVelocityGridKernel::CUDADeformationFieldFromVelocityGridKernel(GlobalContent *conIn, std::string nameIn) : DeformationFieldFromVelocityGridKernel(nameIn)
{
    //cast to the "real type"
    con = dynamic_cast<CudaF3DContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->controlPointImage = con->getCurrentControlPointGrid();
}

void CUDADeformationFieldFromVelocityGridKernel::calculate(bool updateStepNumber) {
    reg_spline_getDefFieldFromVelocityGrid(this->controlPointImage,
                                           this->deformationFieldImage,
                                           updateStepNumber
                                           );
    //4 the moment - to update the gpu
    this->con->setCurrentDeformationField(this->deformationFieldImage);
}
