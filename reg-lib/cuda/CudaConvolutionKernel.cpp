#include "CudaConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
void CudaConvolutionKernel::Calculate(nifti_image *image,
                                      float *sigma,
                                      ConvKernelType kernelType,
                                      int *mask,
                                      bool *timePoint,
                                      bool *axis) {
    // The only caller smooths the pyramid images during registration initialisation, before any
    // device data exists, so the convolution runs on the CPU where the image already lives; a
    // device implementation (Cuda::KernelConvolution) would only add an upload/download round trip
    // for a one-off setup step
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
/* *************************************************************** */
