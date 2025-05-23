#pragma once

#include "Kernel.h"
#include "RNifti.h"

class ResampleImageKernel: public Kernel {
public:
    static std::string GetName() {
        return "ResampleImageKernel";
    }
    ResampleImageKernel() : Kernel() {}
    virtual ~ResampleImageKernel() {}
    virtual void Calculate(int interp, float paddingValue, bool *dtiTimePoint = nullptr, mat33 *jacMat = nullptr) = 0;
};
