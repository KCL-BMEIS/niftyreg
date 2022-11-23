#pragma once

#include "Kernel.h"

class OptimiseKernel : public Kernel{
public:
    static std::string GetName() {
        return "OptimiseKernel";
    }
    OptimiseKernel(std::string name) : Kernel(name) {
    }
    virtual ~OptimiseKernel(){}
    virtual void Calculate(bool affine) = 0;
};
