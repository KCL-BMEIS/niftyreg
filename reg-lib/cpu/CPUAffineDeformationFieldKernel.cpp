#include "CPUAffineDeformationFieldKernel.h"
#include "_reg_globalTrans.h"

void CPUAffineDeformationFieldKernel::calculate(bool compose) {
    reg_affine_getDeformationField(this->affineTransformation,
                                             this->deformationFieldImage,
                                             compose,
                                             this->mask);
}
