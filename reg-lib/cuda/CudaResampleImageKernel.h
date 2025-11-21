#pragma once

#include "ResampleImageKernel.h"
#include "CudaAladinContent.h"

/*
 * kernel functions for image resampling with three interpolation variations
 * */
class CudaResampleImageKernel: public ResampleImageKernel {
public:
    CudaResampleImageKernel(Content *conIn);
    void Calculate(int interp,
                   float paddingValue,
                   bool *dtiTimePoint = nullptr,
                   mat33 *jacMat = nullptr);

private:
    nifti_image *floatingImage;
    nifti_image *warpedImage;

    //cuda ptrs
    float* floatingImageArray_d;
    float* floIJKMat_d;
    float* warpedImageArray_d;
    float* deformationFieldImageArray_d;
    int *mask_d;
};
