#ifndef CPUCONVOLUTIONKERNEL_H
#define CPUCONVOLUTIONKERNEL_H

#include "ConvolutionKernel.h"
#include <string>

class CPUConvolutionKernel : public ConvolutionKernel {
public:
    CPUConvolutionKernel(std::string name);

    void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
};

#endif // CPUCONVOLUTIONKERNEL_H
