#ifndef CLCONVOLUTIONKERNEL_H
#define CLCONVOLUTIONKERNEL_H

#include "ConvolutionKernel.h"
#include "CLContextSingletton.h"

class CLConvolutionKernel : public ConvolutionKernel
{
    public:
       CLConvolutionKernel(std::string name);
       ~CLConvolutionKernel();
       void calculate(nifti_image * image, float *sigma, int kernelType, int *mask = NULL, bool * timePoints = NULL, bool * axis = NULL);
    private:
       CLContextSingletton * sContext;
};

#endif // CLCONVOLUTIONKERNEL_H
