#ifndef CUDACONVOLUTIONKERNEL_H
#define CUDACONVOLUTIONKERNEL_H

#include "ConvolutionKernel.h"
#include "CUDAContextSingletton.h"

//a kernel function for convolution (gaussian smoothing?)
class CUDAConvolutionKernel: public ConvolutionKernel
{
public:

    CUDAConvolutionKernel(std::string name);
    void calculate(nifti_image *image,
                        float *sigma,
                        int kernelType,
                        int *mask = NULL,
                        bool *timePoints = NULL,
                        bool *axis = NULL);

    private:
       //CUDAContextSingletton * cudaSContext;

};

#endif // CUDACONVOLUTIONKERNEL_H
