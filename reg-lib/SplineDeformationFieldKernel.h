#ifndef SPLINEDEFORMATIONFIELDKERNEL_H
#define SPLINEDEFORMATIONFIELDKERNEL_H

#include "Kernel.h"

class SplineDeformationFieldKernel : public Kernel {
public:
    static std::string getName() {
        return "SplineDeformationFieldKernel";
    }

    SplineDeformationFieldKernel( std::string name) : Kernel(name) {
    }

    virtual ~SplineDeformationFieldKernel(){}
    virtual void calculate(bool compose = false) = 0;
};

#endif // SPLINEDEFORMATIONFIELDKERNEL_H
