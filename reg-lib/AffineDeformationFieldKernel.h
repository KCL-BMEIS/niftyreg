#ifndef AFFINEDEFORMATIONFIELDKERNEL_H
#define AFFINEDEFORMATIONFIELDKERNEL_H

#include "Kernel.h"

class AffineDeformationFieldKernel : public Kernel {
public:
    static std::string getName() {
        return "AffineDeformationFieldKernel";
    }

    AffineDeformationFieldKernel( std::string name) : Kernel(name) {
    }

    virtual ~AffineDeformationFieldKernel(){}
    virtual void calculate(bool compose = false) = 0;
};

#endif // AFFINEDEFORMATIONFIELDKERNEL_H
