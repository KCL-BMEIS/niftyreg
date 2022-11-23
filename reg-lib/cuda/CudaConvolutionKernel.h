#pragma once

#include "ConvolutionKernel.h"
#include "CudaContextSingleton.h"

//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel: public ConvolutionKernel
{
public:

    CudaConvolutionKernel(std::string name);
    void Calculate(nifti_image *image,
                        float *sigma,
                        int kernelType,
                        int *mask = nullptr,
                        bool *timePoints = nullptr,
                        bool *axis = nullptr);

    private:
       //CudaContextSingleton * cudaSContext;

};
