#include "CPUConvolutionKernel.h"
#include "_reg_globalTrans.h"

void CPUConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
