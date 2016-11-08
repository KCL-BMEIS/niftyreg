#ifndef CPURESAMPLEIMAGEKERNEL_H
#define CPURESAMPLEIMAGEKERNEL_H

#include "ResampleImageKernel.h"
#include "AladinContent.h"

class CPUResampleImageKernel : public ResampleImageKernel
{
    public:
        CPUResampleImageKernel(AladinContent *con, std::string name);

        nifti_image *floatingImage;
        nifti_image *warpedImage;
        nifti_image *deformationField;
        int *mask;

        void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

#endif // CPURESAMPLEIMAGEKERNEL_H
