#pragma once

#include "ConvolutionKernel.h"
#include <string>

class CpuConvolutionKernel : public ConvolutionKernel {
public:
    CpuConvolutionKernel(std::string name);

    void Calculate(nifti_image *image, float *sigma, int kernelType, int *mask = nullptr, bool *timePoints = nullptr, bool *axis = nullptr);
};
