#include "ClConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClConvolutionKernel::ClConvolutionKernel(std::string name) : ConvolutionKernel(name) {
    sContext = &ClContextSingleton::Instance();
}
/* *************************************************************** */
void ClConvolutionKernel::Calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    //cpu atm
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
/* *************************************************************** */
ClConvolutionKernel::~ClConvolutionKernel() {}
/* *************************************************************** */
