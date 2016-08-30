#ifndef DEFORMATIONFIELDFROMVELOCITYGRIDKERNEL
#define DEFORMATIONFIELDFROMVELOCITYGRIDKERNEL

#include "Kernel.h"

class DeformationFieldFromVelocityGridKernel : public Kernel {
public:
    static std::string getName() {
        return "DeformationFieldFromVelocityGridKernel";
    }

    DeformationFieldFromVelocityGridKernel( std::string name) : Kernel(name) {
    }

    virtual ~DeformationFieldFromVelocityGridKernel(){}
    virtual void calculate(bool updateStepNumber = true) = 0;
};

#endif // DEFORMATIONFIELDFROMVELOCITYGRIDKERNEL

