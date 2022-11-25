#pragma once

#include "Kernel.h"

class AffineDeformationFieldKernel: public Kernel {
public:
    static std::string GetName() {
        return "AffineDeformationFieldKernel";
    }
    AffineDeformationFieldKernel() : Kernel() {}
    virtual ~AffineDeformationFieldKernel() {}
    virtual void Calculate(bool compose = false) = 0;
};
