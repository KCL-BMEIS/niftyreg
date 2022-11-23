#include "CpuAffineDeformationFieldKernel.h"
#include "_reg_globalTrans.h"

CpuAffineDeformationFieldKernel::CpuAffineDeformationFieldKernel(AladinContent *con, std::string nameIn) : AffineDeformationFieldKernel(nameIn) {
    this->deformationFieldImage = con->GetCurrentDeformationField();
    this->affineTransformation = con->GetTransformationMatrix();
    this->mask = con->GetCurrentReferenceMask();
}

void CpuAffineDeformationFieldKernel::Calculate(bool compose) {
   reg_affine_getDeformationField(this->affineTransformation,
                                  this->deformationFieldImage,
                                  compose,
                                  this->mask);
}
