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

    // The content now shares CudaContent's storage: float4 deformation field and compacted mask.
    auto resampleImage = floatingImage->nz > 1 ? NiftyReg::Cuda::ResampleImage<true> : NiftyReg::Cuda::ResampleImage<false>;
    resampleImage(floatingImage,
                  con->GetFloatingImageArray_d(),                 // CudaContent floatingCuda
                  warpedImage,
                  con->GetWarpedImageArray_d(),                   // CudaContent warpedCuda
                  con->Content::GetDeformationField(),            // nifti dims (no device download)
                  con->GetDeformationFieldCuda(),                 // float4 deformation
                  con->GetReferenceMaskCuda(),                    // compacted active-voxel list
                  con->GetActiveVoxelNumber(),
                  interp,
                  paddingValue);
}
/* *************************************************************** */
