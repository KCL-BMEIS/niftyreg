#pragma once

#include "ResampleImageKernel.h"
#include "CudaAladinContent.h"

// Image resampling on CUDA. Thin wrapper over the shared Compute-path resampler Cuda::ResampleImage
// (linear only). Non-linear interpolation is handled on the CPU by reg_aladin (the final cubic warp
// forces the CPU platform, mirroring reg_f3d).
class CudaResampleImageKernel: public ResampleImageKernel {
public:
    CudaResampleImageKernel(Content *conIn);
    void Calculate(int interp,
                   float paddingValue,
                   bool *dtiTimePoint = nullptr,
                   mat33 *jacMat = nullptr);

private:
    CudaAladinContent *con;
    nifti_image *floatingImage;
    nifti_image *warpedImage;
};
