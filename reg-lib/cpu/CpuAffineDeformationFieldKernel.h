#pragma once

#include "AffineDeformationFieldKernel.h"
#include "AladinContent.h"
#include <string>

class CpuAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
    CpuAffineDeformationFieldKernel(Content *conIn);
    void Calculate(bool compose = false);

private:
    mat44 *affineTransformation;
    nifti_image *deformationFieldImage;
    int *mask;
};
