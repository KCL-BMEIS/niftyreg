#ifndef CPUAFFINEDEFORMATIONFIELDKERNEL_H
#define CPUAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"
#include "AladinContent.h"
#include <string>


class CPUAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
        CPUAffineDeformationFieldKernel(AladinContent *con, std::string nameIn);

        void calculate(bool compose = false);

        mat44 *affineTransformation;
        nifti_image *deformationFieldImage;
        int *mask;
};

#endif // AFFINEDEFORMATIONFIELDKERNEL_H
