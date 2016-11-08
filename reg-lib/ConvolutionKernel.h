#ifndef CONVOLUTIONKERNEL_H
#define CONVOLUTIONKERNEL_H

#include "Kernel.h"
#include "nifti1_io.h"

class ConvolutionKernel : public Kernel {
public:
    static std::string getName() {
        return "ConvolutionKernel";
    }
    ConvolutionKernel(std::string name) : Kernel(name) {
    }
    virtual ~ConvolutionKernel(){}
    virtual void calculate(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
};

#endif // CONVOLUTIONKERNEL_H
