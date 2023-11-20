#include "CpuResampleImageKernel.h"
#include "_reg_resampling.h"

/* *************************************************************** */
CpuResampleImageKernel::CpuResampleImageKernel(Content *conIn) : ResampleImageKernel() {
    AladinContent *con = static_cast<AladinContent*>(conIn);
    floatingImage = con->GetFloating();
    warpedImage = con->GetWarped();
    deformationField = con->GetDeformationField();
    mask = con->GetReferenceMask();
}
/* *************************************************************** */
void CpuResampleImageKernel::Calculate(int interp,
                                       float paddingValue,
                                       bool *dtiTimePoint,
                                       mat33 * jacMat) {
    reg_resampleImage(floatingImage,
                      warpedImage,
                      deformationField,
                      mask,
                      interp,
                      paddingValue,
                      dtiTimePoint,
                      jacMat);
}
/* *************************************************************** */
