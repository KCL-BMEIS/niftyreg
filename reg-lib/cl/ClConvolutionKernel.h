#pragma once

#include "ConvolutionKernel.h"
#include "ClContextSingleton.h"

class ClConvolutionKernel: public ConvolutionKernel {
public:
    ClConvolutionKernel() : ConvolutionKernel() {}
    ~ClConvolutionKernel() {}
    void Calculate(nifti_image *image, float *sigma, ConvKernelType kernelType, int *mask = nullptr, bool *timePoints = nullptr, bool *axis = nullptr);
};
