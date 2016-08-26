#include "CPUResampleImageKernel.h"
#include "_reg_resampling.h"

CPUResampleImageKernel::CPUResampleImageKernel(GlobalContent *conIn, std::string name) : ResampleImageKernel(name) {
   this->con = dynamic_cast<GlobalContent*>(conIn);
   floatingImage = this->con->getCurrentFloating();
   warpedImage = this->con->getCurrentWarped();
   deformationField = this->con->getCurrentDeformationField();
   mask = this->con->getCurrentReferenceMask();
}

void CPUResampleImageKernel::calculate(int interp,
                                       float paddingValue,
                                       bool *dti_timepoint,
                                       mat33 * jacMat)
{
   reg_resampleImage(this->floatingImage,
                     this->warpedImage,
                     this->deformationField,
                     this->mask,
                     interp,
                     paddingValue,
                     dti_timepoint,
                     jacMat);
}
