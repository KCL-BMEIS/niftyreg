#pragma once

#include "ResampleImageKernel.h"
#include "AladinContent.h"

class CpuResampleImageKernel : public ResampleImageKernel
{
    public:
        CpuResampleImageKernel(AladinContent *con, std::string name);

        nifti_image *floatingImage;
        nifti_image *warpedImage;
        nifti_image *deformationField;
        int *mask;

        void Calculate(int interp, float paddingValue, bool *dti_timepoint = nullptr, mat33 *jacMat = nullptr);
};
