#include "CPUResampleImageKernel.h"
#include "_reg_resampling.h"

CPUResampleImageKernel::CPUResampleImageKernel(AladinContent *con, std::string name) : ResampleImageKernel( name) {
   floatingImage = con->getCurrentFloating();
   warpedImage = con->getCurrentWarped();
   deformationField = con->getCurrentDeformationField();
   mask = con->getCurrentReferenceMask();
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
