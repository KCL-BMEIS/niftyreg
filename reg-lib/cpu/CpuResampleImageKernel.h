#pragma once

#include "ResampleImageKernel.h"
#include "AladinContent.h"

class CpuResampleImageKernel: public ResampleImageKernel {
public:
    CpuResampleImageKernel(Content *con);
    void Calculate(int interp, float paddingValue, bool *dtiTimePoint = nullptr, mat33 *jacMat = nullptr);

private:
    nifti_image *floatingImage;
    nifti_image *warpedImage;
    nifti_image *deformationField;
    int *mask;
};
