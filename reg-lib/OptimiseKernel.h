#pragma once

#include "Kernel.h"

class OptimiseKernel: public Kernel {
public:
    static std::string GetName() {
        return "OptimiseKernel";
    }
    OptimiseKernel() : Kernel() {}
    virtual ~OptimiseKernel() {}
    virtual void Calculate(bool affine) = 0;
};
