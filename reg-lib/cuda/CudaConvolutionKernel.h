#pragma once

#include "ConvolutionKernel.h"
#include "CudaContext.hpp"

// A kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel: public ConvolutionKernel {
public:
    virtual void Calculate(nifti_image *image,
                           float *sigma,
                           ConvKernelType kernelType,
                           int *mask = nullptr,
                           bool *timePoints = nullptr,
                           bool *axis = nullptr) override;
};
