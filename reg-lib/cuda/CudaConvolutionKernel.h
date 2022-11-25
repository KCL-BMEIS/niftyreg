#pragma once

#include "ConvolutionKernel.h"
#include "CudaContextSingleton.h"

// A kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel: public ConvolutionKernel {
public:
    CudaConvolutionKernel() : ConvolutionKernel() {}
    void Calculate(nifti_image *image,
                   float *sigma,
                   int kernelType,
                   int *mask = nullptr,
                   bool *timePoints = nullptr,
                   bool *axis = nullptr);
};
