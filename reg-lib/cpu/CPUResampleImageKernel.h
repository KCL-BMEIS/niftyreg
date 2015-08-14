#ifndef CPURESAMPLEIMAGEKERNEL_H
#define CPURESAMPLEIMAGEKERNEL_H

#include "ResampleImageKernel.h"
#include "Content.h"

class CPUResampleImageKernel : public ResampleImageKernel
{
    public:
        CPUResampleImageKernel(Content *con, std::string name) : ResampleImageKernel( name) {
            floatingImage = con->getCurrentFloating();
            warpedImage = con->getCurrentWarped();
            deformationField = con->getCurrentDeformationField();
            mask = con->getCurrentReferenceMask();
        }

        nifti_image *floatingImage;
        nifti_image *warpedImage;
        nifti_image *deformationField;
        int *mask;

        void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
};

#endif // CPURESAMPLEIMAGEKERNEL_H
