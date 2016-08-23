#ifndef CPUAFFINEDEFORMATIONFIELDKERNEL_H
#define CPUAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"
#include "AladinContent.h"
#include <string>


class CPUAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
        CPUAffineDeformationFieldKernel(GlobalContent *con, std::string nameIn);
        void calculate(bool compose = false);

private:
        AladinContent *con;
        mat44 *affineTransformation;
        nifti_image *deformationFieldImage;
        int *mask;
};

#endif // AFFINEDEFORMATIONFIELDKERNEL_H
