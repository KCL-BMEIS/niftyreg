#ifndef CPURESAMPLEIMAGEKERNEL_H
#define CPURESAMPLEIMAGEKERNEL_H

#include "ResampleImageKernel.h"
#include "AladinContent.h"

class CPUResampleImageKernel : public ResampleImageKernel
{
public:
        CPUResampleImageKernel(GlobalContent *con, std::string name);
        void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);

private:
        nifti_image *floatingImage;
        nifti_image *warpedImage;
        nifti_image *deformationField;
        int *mask;
        //
        GlobalContent *con;
};

#endif // CPURESAMPLEIMAGEKERNEL_H
