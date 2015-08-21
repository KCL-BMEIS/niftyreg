#include "CudaConvolutionKernel.h"

/* *************************************************************** */
void CudaConvolutionKernel::calculate(nifti_image *image,
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
