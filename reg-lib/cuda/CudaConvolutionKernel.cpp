#include "CudaConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
void CudaConvolutionKernel::Calculate(nifti_image *image,
                                      float *sigma,
                                      ConvKernelType kernelType,
                                      int *mask,
                                      bool *timePoint,
                                      bool *axis) {
    //cpu cheat
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
/* *************************************************************** */
