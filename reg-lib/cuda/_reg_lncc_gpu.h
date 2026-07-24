#pragma once

#include "CudaTools.hpp"
#include "CudaKernelConvolution.hpp"
#include "_reg_measure_gpu.h"
#include "_reg_lncc.h"

/* *************************************************************** */
/// @brief LNCC measure of similarity class on the device
class reg_lncc_gpu: public reg_lncc, public reg_measure_gpu {
public:
    /// @brief reg_lncc_gpu class constructor
    reg_lncc_gpu();
    /// @brief reg_lncc_gpu class destructor
    virtual ~reg_lncc_gpu();

    /// @brief Initialise the reg_lncc_gpu object
    // Bring the CPU base overload into scope; the GPU override below intentionally adds a second overload
    using reg_measure::InitialiseMeasure;
    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
                                   int *refMask,
                                   int *refMaskCuda,
                                   size_t activeVoxNum,
                                   nifti_image *warpedImg,
                                   float *warpedImgCuda,
                                   nifti_image *warpedGrad,
                                   float4 *warpedGradCuda,
                                   nifti_image *voxelBasedGrad,
                                   float4 *voxelBasedGradCuda,
                                   nifti_image *localWeightSim = nullptr,
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr,
                                   size_t activeVoxNumBw = 0) override;
    /// @brief Returns the lncc value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the lncc value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based lncc gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based lncc gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;

protected:
    // Device scratch buffers mirroring the CPU reg_lncc images. The four local statistics are
    // packed into the lanes of ONE float4 image (x = mean, y = sdev, z = warpedMean,
    // w = warpedSdev) so the shared Cuda::KernelConvolutionPacked smooths all four in a single
    // pass with fully coalesced accesses; per lane the maths is identical to convolving each
    // image alone, preserving bit-exactness with the CPU. The correlation image uses lane .x
    // (the gradient reuses its lanes x/y/z for the three temp fields it smooths).
    thrust::device_vector<float4> statsImageCuda;
    thrust::device_vector<float4> correlationImageCuda;
    // Per-voxel masks: the base mask is the host reg_measure mask uploaded ONCE at initialise
    // time; the combined mask (base minus every voxel NaN in the reference or warped image) is
    // rebuilt on device each call without any host transfer.
    thrust::device_vector<int> baseMaskCuda;
    thrust::device_vector<int> forwardMaskCuda;

    thrust::device_vector<float4> statsImageBwCuda;
    thrust::device_vector<float4> correlationImageBwCuda;
    thrust::device_vector<int> baseMaskBwCuda;
    thrust::device_vector<int> backwardMaskCuda;

    // Reusable convolution scratch: avoids per-call cudaMalloc/cudaFree and lets the
    // convolutions of one similarity value share a single smoothed-density computation, since they
    // all use the same combined mask (which already excludes every NaN voxel).
    NiftyReg::Cuda::KernelConvolutionWorkspace convWorkspace;
};
/* *************************************************************** */
