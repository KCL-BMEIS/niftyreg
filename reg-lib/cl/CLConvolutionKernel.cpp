#include "CLConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
CLConvolutionKernel::CLConvolutionKernel(std::string name) : ConvolutionKernel(name) {
    sContext = &CLContextSingletton::Instance();
}
/* *************************************************************** */
void CLConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    //cpu atm
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
/* *************************************************************** */
CLConvolutionKernel::~CLConvolutionKernel() {}
/* *************************************************************** */
