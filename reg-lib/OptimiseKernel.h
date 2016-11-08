#ifndef OPTIMISEKERNEL_H
#define OPTIMISEKERNEL_H

#include "Kernel.h"

class OptimiseKernel : public Kernel{
public:
    static std::string getName() {
        return "OptimiseKernel";
    }
    OptimiseKernel(std::string name) : Kernel(name) {
    }
    virtual ~OptimiseKernel(){}
    virtual void calculate(bool affine) = 0;
};

#endif // OPTIMISEKERNEL_H
