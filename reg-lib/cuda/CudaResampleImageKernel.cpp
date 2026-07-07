#include "CudaResampleImageKernel.h"
#include "CudaResampling.hpp"

/* *************************************************************** */
CudaResampleImageKernel::CudaResampleImageKernel(Content *conIn) : ResampleImageKernel() {
    con = dynamic_cast<CudaAladinContent*>(conIn);
    floatingImage = con->AladinContent::GetFloating();
    warpedImage = con->AladinContent::GetWarped();

    if (floatingImage->datatype != warpedImage->datatype)
        NR_FATAL_ERROR("Floating and warped images should have the same data type");
    if (floatingImage->nt != warpedImage->nt)
        NR_FATAL_ERROR("Floating and warped images have different dimensions along the time axis");
}
/* *************************************************************** */
void CudaResampleImageKernel::Calculate(int interp,
                                        float paddingValue,
                                        bool *dtiTimePoint,
                                        mat33 *jacMat) {
    if (dtiTimePoint != nullptr || jacMat != nullptr)
        NR_FATAL_ERROR("DTI resampling is not supported on the GPU");
    if (interp != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported on the GPU");

    auto resampleImage = floatingImage->nz > 1 ? NiftyReg::Cuda::ResampleImage<true> : NiftyReg::Cuda::ResampleImage<false>;
    resampleImage(floatingImage,
                  con->GetFloatingCuda(),
                  warpedImage,
                  con->GetWarpedCuda(),
                  con->Content::GetDeformationField(),
                  con->GetDeformationFieldCuda(),
                  con->GetReferenceMaskCuda(),
                  con->GetActiveVoxelNumber(),
                  interp,
                  paddingValue);
}
/* *************************************************************** */
