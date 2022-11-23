#pragma once

#include "AffineDeformationFieldKernel.h"
#include "AladinContent.h"
#include <string>

class CpuAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
        CpuAffineDeformationFieldKernel(AladinContent *con, std::string nameIn);

        void Calculate(bool compose = false);

        mat44 *affineTransformation;
        nifti_image *deformationFieldImage;
        int *mask;
};
