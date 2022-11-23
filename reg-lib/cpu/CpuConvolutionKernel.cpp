#include "CpuConvolutionKernel.h"
#include "_reg_globalTrans.h"

CpuConvolutionKernel::CpuConvolutionKernel(std::string name) : ConvolutionKernel(name) {
}

void CpuConvolutionKernel::Calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis) {
    reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
