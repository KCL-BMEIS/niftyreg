#pragma once

#include "Kernel.h"
#include "nifti1_io.h"

class ResampleImageKernel : public Kernel {
public:
    static std::string GetName() {
        return "ResampleImageKernel";
    }
    ResampleImageKernel( std::string name) : Kernel(name) {
    }

    virtual ~ResampleImageKernel(){}

    virtual void Calculate(int interp, float paddingValue, bool *dti_timepoint = nullptr, mat33 *jacMat = nullptr) = 0;
};
