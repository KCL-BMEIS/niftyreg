#pragma once

#include "ConvolutionKernel.h"
#include <string>

class CpuConvolutionKernel: public ConvolutionKernel {
public:
    CpuConvolutionKernel() : ConvolutionKernel() {}
    void Calculate(nifti_image *image, float *sigma, ConvKernelType kernelType, int *mask = nullptr, bool *timePoints = nullptr, bool *axis = nullptr);
};
