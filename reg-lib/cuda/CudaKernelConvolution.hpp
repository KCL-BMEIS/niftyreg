#pragma once

#include "CudaTools.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
/** @brief Smooth an image using a specified kernel
 * @param image Image to be smoothed
 * @param imageCuda Image to be smoothed
 * @param sigma Standard deviation of the kernel to use.
 * The kernel is bounded between +/- 3 sigma.
 * @param kernelType Type of kernel to use.
 * @param timePoints Boolean array to specify which time points have to be
 * smoothed. The array follow the dim array of the nifti header.
 * @param axes Boolean array to specify which axes have to be
 * smoothed. The array follow the dim array of the nifti header.
 */
template<ConvKernelType kernelType>
void KernelConvolution(const nifti_image *image,
                       float4 *imageCuda,
                       const float *sigma,
                       const bool *timePoints = nullptr,
                       const bool *axes = nullptr);
/* *************************************************************** */
}
/* *************************************************************** */
