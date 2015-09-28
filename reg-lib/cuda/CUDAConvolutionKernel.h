#ifndef CUDACONVOLUTIONKERNEL_H
#define CUDACONVOLUTIONKERNEL_H

#include "ConvolutionKernel.h"

//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel: public ConvolutionKernel
{
public:

    CudaConvolutionKernel(std::string name) : ConvolutionKernel(name) {}
    void calculate(nifti_image *image,
                        float *sigma,
                        int kernelType,
                        int *mask = NULL,
                        bool *timePoints = NULL,
                        bool *axis = NULL);

};

#endif // CUDACONVOLUTIONKERNEL_H
