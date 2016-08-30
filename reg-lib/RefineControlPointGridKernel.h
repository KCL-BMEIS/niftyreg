#ifndef REFINECONTROLPOINTGRIDKERNEL
#define REFINECONTROLPOINTGRIDKERNEL

#include "Kernel.h"

class RefineControlPointGridKernel : public Kernel {
public:
    static std::string getName() {
        return "RefineControlPointGridKernel";
    }

    RefineControlPointGridKernel( std::string name) : Kernel(name) {
    }

    virtual ~RefineControlPointGridKernel(){}
    virtual void calculate() = 0;
};

#endif // REFINECONTROLPOINTGRIDKERNEL

