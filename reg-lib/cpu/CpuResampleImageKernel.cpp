#include "CpuResampleImageKernel.h"
#include "_reg_resampling.h"

CpuResampleImageKernel::CpuResampleImageKernel(AladinContent *con, std::string name) : ResampleImageKernel( name) {
   floatingImage = con->GetCurrentFloating();
   warpedImage = con->GetCurrentWarped();
   deformationField = con->GetCurrentDeformationField();
   mask = con->GetCurrentReferenceMask();
}

void CpuResampleImageKernel::Calculate(int interp,
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
