#pragma once

#include "CudaTools.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
/** @brief Reusable device scratch for KernelConvolution.
 *
 * Passing the same workspace to repeated convolutions avoids re-allocating the internal
 * buffers on every call - a significant host-side (cudaMalloc/cudaFree) overhead when a routine
 * convolves many images per iteration (e.g. LNCC runs five convolutions per similarity value).
 *
 * It can additionally cache the smoothed density field: when @c densityValid is true the next
 * convolution reuses the cached @c density / @c nanImage instead of recomputing them. The caller
 * is responsible for only keeping @c densityValid true across convolutions that genuinely share
 * the same density - i.e. the same mask, with every in-mask voxel finite in every image. The LNCC
 * scratch images qualify (their combined mask already excludes every NaN voxel), so the density is
 * identical for all of them and need only be computed once.
 */
struct KernelConvolutionWorkspace {
    // The weighted (Gaussian/Cubic/Linear, and 2D-Mean) path is voxel-parallel and ping-pongs the
    // intensity between intensityA/intensityB and the density between density/densityAux (the
    // buffer holding the final smoothed density is recorded in densityInAux). The cumulative
    // (3D-Mean) path is line-parallel and in-place, reusing intensityA/intensityB as its per-plane
    // intensity/density scratch.
    thrust::device_vector<float> density;
    thrust::device_vector<float> densityAux;
    thrust::device_vector<bool> nanImage;
    thrust::device_vector<float> intensityA;
    thrust::device_vector<float> intensityB;
    thrust::device_vector<float> kernel;
    bool densityValid = false;
    /// True when the cached smoothed density lives in densityAux rather than density (the weighted
    /// path ping-pongs between the two, so with an odd number of smoothed axes it ends in the aux
    /// buffer; tracking the location avoids a whole-volume copy per density computation).
    bool densityInAux = false;
    // Key of the kernel weights currently uploaded in `kernel`, so consecutive convolutions with
    // the same kernel (as in LNCC, where every convolution of a call shares sigma and type) skip
    // the per-axis host-to-device upload. cachedKernelKind is int(ConvKernelType) or -1.
    int cachedKernelKind = -1;
    int cachedKernelRadius = -1;
    double cachedKernelTemp = -1;
    /// Grow the buffers so the intensity ping-pong holds at least @p voxelNumber * @p channels
    /// floats and the density/NaN buffers hold @p voxelNumber elements (never shrinks).
    /// Defined in CudaKernelConvolution.cu so device_vector::resize is only ever instantiated by
    /// the device compiler (host translation units that merely see this type must not trigger it).
    void EnsureSize(const size_t voxelNumber, const int channels = 1);
};
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
 * @param maskCuda Optional per-voxel mask (device pointer). When provided, a voxel
 * contributes to the convolution only if maskCuda[voxel] >= 0, mirroring the CPU
 * reg_tools_kernelConvolution density behaviour. When nullptr (default) all
 * non-NaN voxels are active, preserving the previous behaviour.
 * @param workspace Optional reusable scratch (see KernelConvolutionWorkspace). When nullptr the
 * buffers are allocated locally for this call and the density is always recomputed (the default,
 * unchanged behaviour). When provided, the buffers are reused and the smoothed density may be
 * cached/reused across calls (controlled by workspace->densityValid).
 *
 * @tparam AccType Precision of the per-line weighted-sum accumulation. Defaults to double (matching
 * the CPU reg_tools_kernelConvolution, so the result is bit-exact with it). LNCC uses float to move
 * the accumulation off the (slow on some GPUs) FP64 pipe
 */
template<ConvKernelType kernelType, class AccType = double>
void KernelConvolution(const nifti_image *image,
                       float4 *imageCuda,
                       const float *sigma,
                       const bool *timePoints = nullptr,
                       const bool *axes = nullptr,
                       const int *maskCuda = nullptr,
                       KernelConvolutionWorkspace *workspace = nullptr);
/* *************************************************************** */
/** @brief Smooth all four channels of a float4 image in a single pass.
 *
 * Same maths as KernelConvolution applied to each float4 component independently: per voxel and
 * per channel the tap order, accumulation type and normalisation are identical to convolving that
 * channel alone, so the per-channel results are bit-for-bit the same as four single-channel calls.
 * Packing four images into one float4 volume and using this function instead of four calls
 * quarters the kernel launches and the elementwise init/normalise passes, and turns the strided
 * one-float-in-sixteen-bytes accesses of a `.x`-only image into fully coalesced float4 accesses.
 *
 * Requirements (all satisfied by the LNCC scratch images, the only caller):
 *  - @p image describes the 3D geometry only; every channel is smoothed with sigma[0].
 *  - The mask/NaN pattern must be identical across channels (the shared density is derived from
 *    channel .x): every in-mask voxel must be finite in every channel. Out-of-mask lanes may hold
 *    any garbage - lanes never mix.
 *  - @p workspace is mandatory (it provides the float4-sized scratch and the density cache).
 *
 * The cumulative 3D-Mean filter cannot be voxel-parallelised, so for that case this function
 * falls back to the (bit-identical) single-channel path applied per channel.
 */
template<ConvKernelType kernelType, class AccType = double>
void KernelConvolutionPacked(const nifti_image *image,
                             float4 *imageCuda,
                             const float *sigma,
                             const int *maskCuda,
                             KernelConvolutionWorkspace *workspace);
/* *************************************************************** */
}
/* *************************************************************** */
