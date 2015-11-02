#include "CUDAConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
CUDAConvolutionKernel::CUDAConvolutionKernel(std::string name) : ConvolutionKernel(name) {
    //cudaSContext = &CUDAContextSingletton::Instance();
}
/* *************************************************************** */
void CUDAConvolutionKernel::calculate(nifti_image *image,
                                                  float *sigma,
                                                  int kernelType,
                                                  int *mask,
                                                  bool *timePoint,
                                                  bool *axis)
{
    //cpu cheat
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
/* *************************************************************** */
