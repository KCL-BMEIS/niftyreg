#pragma once

#include "Kernel.h"
#include "nifti1_io.h"

class ConvolutionKernel : public Kernel {
public:
    static std::string GetName() {
        return "ConvolutionKernel";
    }
    ConvolutionKernel(std::string name) : Kernel(name) {
    }
    virtual ~ConvolutionKernel(){}
    virtual void Calculate(nifti_image *image, float *sigma, int kernelType, int *mask = nullptr, bool *timePoints = nullptr, bool *axis = nullptr) = 0;
};
