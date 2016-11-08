#include "CPUConvolutionKernel.h"
#include "_reg_globalTrans.h"

CPUConvolutionKernel::CPUConvolutionKernel(std::string name) : ConvolutionKernel(name) {
}

void CPUConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
