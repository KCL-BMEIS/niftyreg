#ifndef CPUAFFINEDEFORMATIONFIELDKERNEL_H
#define CPUAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"
#include "Content.h"
#include <string>


class CPUAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
        CPUAffineDeformationFieldKernel(Content *con, std::string nameIn) : AffineDeformationFieldKernel(nameIn) {
            this->deformationFieldImage = con->getCurrentDeformationField();
            this->affineTransformation = con->getTransformationMatrix();
            this->mask = con->getCurrentReferenceMask();
        }


        void calculate(bool compose = false);

        mat44 *affineTransformation;
        nifti_image *deformationFieldImage;
        int *mask;
};

#endif // AFFINEDEFORMATIONFIELDKERNEL_H
