#include "ClConvolutionKernel.h"
#include "_reg_tools.h"

/* *************************************************************** */
void ClConvolutionKernel::Calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
/* *************************************************************** */
