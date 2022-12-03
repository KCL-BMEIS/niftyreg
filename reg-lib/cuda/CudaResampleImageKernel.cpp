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

    if (floatingImage->datatype != warpedImage->datatype) {
        reg_print_fct_error("CudaResampleImageKernel::CudaResampleImageKernel");
        reg_print_msg_error("Floating and warped images should have the same data type. Exit.");
        reg_exit();
    }

    if (floatingImage->nt != warpedImage->nt) {
        reg_print_fct_error("CudaResampleImageKernel::CudaResampleImageKernel");
        reg_print_msg_error("Floating and warped images have different dimension along the time axis. Exit.");
        reg_exit();
    }
}
/* *************************************************************** */
void CudaResampleImageKernel::Calculate(int interp,
                                        float paddingValue,
                                        bool *dti_timepoint,
                                        mat33 * jacMat) {
    launchResample(floatingImage,
                   warpedImage,
                   interp,
                   paddingValue,
                   dti_timepoint,
                   jacMat,
                   &floatingImageArray_d,
                   &warpedImageArray_d,
                   &deformationFieldImageArray_d,
                   &mask_d,
                   &floIJKMat_d);
}
/* *************************************************************** */