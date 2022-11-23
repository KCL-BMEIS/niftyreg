#pragma once

#include "ConvolutionKernel.h"
#include "ClContextSingleton.h"

class ClConvolutionKernel : public ConvolutionKernel
{
    public:
       ClConvolutionKernel(std::string name);
       ~ClConvolutionKernel();
       void Calculate(nifti_image * image, float *sigma, int kernelType, int *mask = nullptr, bool * timePoints = nullptr, bool * axis = nullptr);
    private:
       ClContextSingleton * sContext;
};
