#include "CudaConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
CudaConvolutionKernel::CudaConvolutionKernel(std::string name) : ConvolutionKernel(name) {
    //cudaSContext = &CudaContextSingleton::Instance();
}
/* *************************************************************** */
void CudaConvolutionKernel::Calculate(nifti_image *image,
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
