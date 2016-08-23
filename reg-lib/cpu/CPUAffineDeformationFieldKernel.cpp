#include "CPUAffineDeformationFieldKernel.h"
#include "_reg_globalTrans.h"

CPUAffineDeformationFieldKernel::CPUAffineDeformationFieldKernel(GlobalContent *conIn, std::string nameIn) : AffineDeformationFieldKernel(nameIn) {
    //cast to the "real type"
    con = dynamic_cast<AladinContent*>(conIn);
    this->deformationFieldImage = con->getCurrentDeformationField();
    this->affineTransformation = con->getTransformationMatrix();
    this->mask = con->getCurrentReferenceMask();
}

void CPUAffineDeformationFieldKernel::calculate(bool compose) {
   reg_affine_getDeformationField(this->affineTransformation,
                                  this->deformationFieldImage,
                                  compose,
                                  this->mask);
}
