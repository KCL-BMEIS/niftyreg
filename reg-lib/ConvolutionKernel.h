#pragma once

#include "Kernel.h"
#include "_reg_tools.h"

class ConvolutionKernel: public Kernel {
public:
    static std::string GetName() {
        return "ConvolutionKernel";
    }
    ConvolutionKernel() : Kernel() {}
    virtual ~ConvolutionKernel() {}
    virtual void Calculate(nifti_image *image, float *sigma, ConvKernelType kernelType, int *mask = nullptr, bool *timePoints = nullptr, bool *axis = nullptr) = 0;
};
