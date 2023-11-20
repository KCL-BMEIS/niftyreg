#include "CudaResampleImageKernel.h"
#include "resampleKernel.h"

/* *************************************************************** */
CudaResampleImageKernel::CudaResampleImageKernel(Content *conIn) : ResampleImageKernel() {
    CudaAladinContent *con = static_cast<CudaAladinContent*>(conIn);

    floatingImage = con->AladinContent::GetFloating();
    warpedImage = con->AladinContent::GetWarped();

    //cuda ptrs
    floatingImageArray_d = con->GetFloatingImageArray_d();
    warpedImageArray_d = con->GetWarpedImageArray_d();
    deformationFieldImageArray_d = con->GetDeformationFieldArray_d();
    mask_d = con->GetMask_d();
    floIJKMat_d = con->GetFloIJKMat_d();

    if (floatingImage->datatype != warpedImage->datatype)
        NR_FATAL_ERROR("Floating and warped images should have the same data type");

    if (floatingImage->nt != warpedImage->nt)
        NR_FATAL_ERROR("Floating and warped images have different dimensions along the time axis");
}
/* *************************************************************** */
void CudaResampleImageKernel::Calculate(int interp,
                                        float paddingValue,
                                        bool *dtiTimePoint,
                                        mat33 * jacMat) {
    launchResample(floatingImage,
                   warpedImage,
                   interp,
                   paddingValue,
                   dtiTimePoint,
                   jacMat,
                   &floatingImageArray_d,
                   &warpedImageArray_d,
                   &deformationFieldImageArray_d,
                   &mask_d,
                   &floIJKMat_d);
}
/* *************************************************************** */
