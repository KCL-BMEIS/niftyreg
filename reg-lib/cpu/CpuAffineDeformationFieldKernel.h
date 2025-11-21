#pragma once

#include "AffineDeformationFieldKernel.h"
#include "AladinContent.h"
#include <string>

class CpuAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
    CpuAffineDeformationFieldKernel(Content *conIn);
    virtual void Calculate(bool compose = false) override;

private:
    mat44 *affineTransformation;
    nifti_image *deformationFieldImage;
    int *mask;
};
