/*
 * @file _reg_lncc_gpu.cu
 * @brief CUDA implementation of the LNCC similarity measure
 * Copyright (c) 2026, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#include "_reg_lncc_gpu.h"
#include "CudaKernelConvolution.hpp"

/* *************************************************************** */
// Convolve a single-channel float4 buffer using the .x component
static void ConvolveSingleChannel(const nifti_image *img, float4 *buf, float sigma, ConvKernelType kType) {
    bool tp[] = { true, false, false, false };
    switch (kType) {
    case ConvKernelType::Gaussian:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Gaussian>(img, buf, &sigma, tp);
        break;
    case ConvKernelType::Mean:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Mean>(img, buf, &sigma, tp);
        break;
    case ConvKernelType::Cubic:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Cubic>(img, buf, &sigma, tp);
        break;
    case ConvKernelType::Linear:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Linear>(img, buf, &sigma, tp);
        break;
    default:
        NR_FATAL_ERROR("Unknown kernel type for LNCC GPU convolution");
    }
}
/* *************************************************************** */
// Compute local mean (E[x]) and local standard deviation (sqrt(E[x^2]-E[x]^2))
// for both the reference slice and the warped slice.
// Inactive voxels (mask == -1 or NaN) are set to NaN so the convolution
// infrastructure excludes them from neighbourhood sums.
// After this call:
//   meanBuf.x   = E[ref]      sdevBuf.x   = sdev_ref
//   warpedMeanBuf.x = E[war]  warpedSdevBuf.x = sdev_war
void UpdateLocalStatImagesCuda(const nifti_image *referenceImage,
                                      const float *referenceImageCuda,
                                      const float *warpedImageCuda,
                                      const int *maskCuda,
                                      float4 *meanBuf,
                                      float4 *sdevBuf,
                                      float4 *warpedMeanBuf,
                                      float4 *warpedSdevBuf,
                                      float kernelSigma,
                                      ConvKernelType kernelType,
                                      int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const float *refSlice = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warSlice = warpedImageCuda + currentTimePoint * voxelNumber;

    // Pack values into .x of float4; use NaN for inactive/NaN voxels
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
        [=] __device__(size_t i) {
            const float refVal = refSlice[i];
            const float warVal = warSlice[i];
            const bool active = (maskCuda[i] != -1) && (refVal == refVal) && (warVal == warVal);
            constexpr float nan = std::numeric_limits<float>::quiet_NaN();
            meanBuf[i]       = make_float4(active ? refVal           : nan, 0.f, 0.f, 0.f);
            sdevBuf[i]       = make_float4(active ? refVal * refVal  : nan, 0.f, 0.f, 0.f);
            warpedMeanBuf[i] = make_float4(active ? warVal           : nan, 0.f, 0.f, 0.f);
            warpedSdevBuf[i] = make_float4(active ? warVal * warVal  : nan, 0.f, 0.f, 0.f);
        });

    // Convolve: E[ref], E[ref^2], E[war], E[war^2]
    ConvolveSingleChannel(referenceImage, meanBuf,       kernelSigma, kernelType);
    ConvolveSingleChannel(referenceImage, sdevBuf,       kernelSigma, kernelType);
    ConvolveSingleChannel(referenceImage, warpedMeanBuf, kernelSigma, kernelType);
    ConvolveSingleChannel(referenceImage, warpedSdevBuf, kernelSigma, kernelType);

    // sdev = sqrt(max(0, E[x^2] - E[x]^2)), values below epsilon clamped to 0
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
        [=] __device__(size_t i) {
            const float mean = meanBuf[i].x;
            if (mean == mean) {
                float v = sdevBuf[i].x - mean * mean;
                v = v > 0.f ? sqrtf(v) : 0.f;
                sdevBuf[i].x = v < 1e-6f ? 0.f : v;
            }
            const float warMean = warpedMeanBuf[i].x;
            if (warMean == warMean) {
                float v = warpedSdevBuf[i].x - warMean * warMean;
                v = v > 0.f ? sqrtf(v) : 0.f;
                warpedSdevBuf[i].x = v < 1e-6f ? 0.f : v;
            }
        });
}
/* *************************************************************** */
// Compute the LNCC value for one time point.
// Requires meanBuf/sdevBuf/warpedMeanBuf/warpedSdevBuf to be pre-filled
// by UpdateLocalStatImagesCuda.  correlationBuf is used as a scratch buffer.
// Returns sum(|lncc|) / activeCount.
double GetLnccValueCuda(const nifti_image *referenceImage,
                               const float *referenceImageCuda,
                               const float *warpedImageCuda,
                               const int *maskCuda,
                               const float4 *meanBuf,
                               const float4 *sdevBuf,
                               const float4 *warpedMeanBuf,
                               const float4 *warpedSdevBuf,
                               float4 *correlationBuf,
                               float kernelSigma,
                               ConvKernelType kernelType,
                               int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const float *refSlice = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warSlice = warpedImageCuda + currentTimePoint * voxelNumber;

    // Initialise correlation as ref * war (NaN for inactive voxels)
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
        [=] __device__(size_t i) {
            const float refVal = refSlice[i];
            const float warVal = warSlice[i];
            const bool active = (maskCuda[i] != -1) && (refVal == refVal) && (warVal == warVal);
            correlationBuf[i] = make_float4(
                active ? refVal * warVal : std::numeric_limits<float>::quiet_NaN(),
                0.f, 0.f, 0.f);
        });

    // Convolve to get E[ref * war]
    ConvolveSingleChannel(referenceImage, correlationBuf, kernelSigma, kernelType);

    // Reduce: sum |lncc| and count active voxels
    const auto result = thrust::transform_reduce(thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(voxelNumber),
        [=] __device__(size_t i) -> double2 {
            const float corr = correlationBuf[i].x;
            if (corr != corr) return {};   // NaN → inactive voxel

            const float sdev_ref = sdevBuf[i].x;
            const float sdev_war = warpedSdevBuf[i].x;
            if (sdev_ref == 0.f || sdev_war == 0.f) return {};

            const double lncc = (corr - (double)meanBuf[i].x * warpedMeanBuf[i].x)
                                 / ((double)sdev_ref * sdev_war);
            if (lncc != lncc || isinf(lncc)) return {};
            return { fabs(lncc), 1.0 };
        },
        make_double2(0.0, 0.0),
        thrust::plus<double2>());

    if (result.y == 0.0) return 0.0;
    return result.x / result.y;
}
/* *************************************************************** */
// Compute the voxel-based LNCC gradient for one time point.
// Requires meanBuf/sdevBuf (from UpdateLocalStatImagesCuda) to be valid.
// warpedMeanBuf, warpedSdevBuf, and correlationBuf are used as scratch and
// will be overwritten.
void GetVoxelBasedLnccGradientCuda(const nifti_image *referenceImage,
                                          const float *referenceImageCuda,
                                          const float *warpedImageCuda,
                                          const int *maskCuda,
                                          const float4 *meanBuf,
                                          const float4 *sdevBuf,
                                          float4 *warpedMeanBuf,
                                          float4 *warpedSdevBuf,
                                          float4 *correlationBuf,
                                          const float4 *warpedGradCuda,
                                          float4 *voxelBasedGradCuda,
                                          float kernelSigma,
                                          ConvKernelType kernelType,
                                          int currentTimePoint,
                                          double timePointWeight) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const float *refSlice = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warSlice = warpedImageCuda + currentTimePoint * voxelNumber;

    // Step 1: init correlation as ref * war, convolve → E[ref * war]
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
        [=] __device__(size_t i) {
            const float refVal = refSlice[i];
            const float warVal = warSlice[i];
            const bool active = (maskCuda[i] != -1) && (refVal == refVal) && (warVal == warVal);
            correlationBuf[i] = make_float4(
                active ? refVal * warVal : std::numeric_limits<float>::quiet_NaN(),
                0.f, 0.f, 0.f);
        });
    ConvolveSingleChannel(referenceImage, correlationBuf, kernelSigma, kernelType);

    // Step 2: compute intermediate derivative terms for each active voxel and count them.
    // The buffers warpedMeanBuf, warpedSdevBuf, correlationBuf are reused:
    //   warpedMeanBuf.x → temp1  (= ∂|lncc|/∂E[war])
    //   warpedSdevBuf.x → temp2  (= ∂|lncc|/∂sdev_war)
    //   correlationBuf.x → temp3 (= ∂|lncc|/∂E[ref*war])
    // Inactive voxels (outside the mask or NaN) are set to NaN so the subsequent
    // convolution treats them as missing data, whereas active voxels with
    // degenerate stats are set to zero so they remain part of the convolution
    // (matching the CPU reference).
    const size_t activeVoxelNumber = thrust::transform_reduce(thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(voxelNumber),
        [=] __device__(size_t i) -> size_t {
            const float refVal = refSlice[i];
            const float warVal = warSlice[i];
            const bool active = (maskCuda[i] != -1) && (refVal == refVal) && (warVal == warVal);

            if (!active) {
                constexpr float nan = std::numeric_limits<float>::quiet_NaN();
                warpedMeanBuf[i].x  = nan;
                warpedSdevBuf[i].x  = nan;
                correlationBuf[i].x = nan;
                return size_t(0);
            }

            // Read current stats (warpedMeanBuf and warpedSdevBuf still hold E[war] and sdev_war)
            const double refMean  = meanBuf[i].x;
            const double warMean  = warpedMeanBuf[i].x;
            const double sdevRef  = sdevBuf[i].x;
            const double sdevWar  = warpedSdevBuf[i].x;
            const double corrVal  = correlationBuf[i].x - refMean * warMean; // local cross-covariance

            const double sRW  = sdevRef * sdevWar;           // sdev_ref * sdev_war
            const double sRW3 = sdevRef * sdevWar * sdevWar * sdevWar;

            // Use direct division; inf/NaN from zero sdev is caught by the validity check below
            double temp1 = 1.0 / sRW;
            double temp2 = corrVal / sRW3;
            double temp3 = corrVal * warMean / sRW3 - refMean / sRW;

            if (temp1 != temp1 || isinf(temp1) ||
                temp2 != temp2 || isinf(temp2) ||
                temp3 != temp3 || isinf(temp3)) {
                // Active voxel with degenerate stats (e.g. local sdev ~ 0).
                // Matching the CPU reference, keep it in the subsequent convolution
                // as zero (so it still contributes to neighbouring voxels and
                // receives a smoothed gradient) but do not count it as active.
                warpedMeanBuf[i].x  = 0.f;
                warpedSdevBuf[i].x  = 0.f;
                correlationBuf[i].x = 0.f;
                return size_t(0);
            }

            // Derivative of the absolute value function
            if (corrVal < 0.0) { temp1 = -temp1; temp2 = -temp2; temp3 = -temp3; }

            warpedMeanBuf[i].x  = static_cast<float>(temp1);
            warpedSdevBuf[i].x  = static_cast<float>(temp2);
            correlationBuf[i].x = static_cast<float>(temp3);
            return size_t(1);
        },
        size_t(0),
        thrust::plus<size_t>());

    if (activeVoxelNumber == 0) return;
    const double adjustedWeight = timePointWeight / static_cast<double>(activeVoxelNumber);

    // Step 3: smooth the three derivative buffers
    ConvolveSingleChannel(referenceImage, warpedMeanBuf,  kernelSigma, kernelType);
    ConvolveSingleChannel(referenceImage, warpedSdevBuf,  kernelSigma, kernelType);
    ConvolveSingleChannel(referenceImage, correlationBuf, kernelSigma, kernelType);

    // Step 4: accumulate the gradient
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
        [=] __device__(size_t i) {
            const float temp1 = warpedMeanBuf[i].x;
            if (temp1 != temp1) return;  // NaN → inactive voxel

            const float4 spa = warpedGradCuda[i];
            if (spa.x != spa.x || spa.y != spa.y || spa.z != spa.z) return;

            const double common =
                (temp1 * refSlice[i]
                 - warpedSdevBuf[i].x * warSlice[i]
                 + correlationBuf[i].x) * adjustedWeight;

            float4 grad = voxelBasedGradCuda[i];
            grad.x -= static_cast<float>(common * spa.x);
            grad.y -= static_cast<float>(common * spa.y);
            grad.z -= static_cast<float>(common * spa.z);
            voxelBasedGradCuda[i] = grad;
        });
}
/* *************************************************************** */
reg_lncc_gpu::reg_lncc_gpu(): reg_lncc::reg_lncc() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_lncc_gpu::~reg_lncc_gpu() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_lncc_gpu::InitialiseMeasure(nifti_image *refImg, float *refImgCuda,
                                     nifti_image *floImg, float *floImgCuda,
                                     int *refMask, int *refMaskCuda,
                                     size_t activeVoxNum,
                                     nifti_image *warpedImg, float *warpedImgCuda,
                                     nifti_image *warpedGrad, float4 *warpedGradCuda,
                                     nifti_image *voxelBasedGrad, float4 *voxelBasedGradCuda,
                                     nifti_image *localWeightSim, float *localWeightSimCuda,
                                     int *floMask, int *floMaskCuda,
                                     nifti_image *warpedImgBw, float *warpedImgBwCuda,
                                     nifti_image *warpedGradBw, float4 *warpedGradBwCuda,
                                     nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda) {
    // CPU initialisation: rescales images to [0,1] and allocates CPU buffers
    reg_lncc::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                                localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    // Store GPU pointers
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda,
                                       refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda,
                                       voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda,
                                       floMask, floMaskCuda,
                                       warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda,
                                       voxelBasedGradBw, voxelBasedGradBwCuda);

    // Reflect the [0,1] rescaling performed by reg_lncc::InitialiseMeasure on the GPU
    Cuda::TransferNiftiToDevice(this->referenceImageCuda, this->referenceImage);
    Cuda::TransferNiftiToDevice(this->floatingImageCuda, this->floatingImage);

    // Allocate intermediate forward-pass GPU buffers
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);
    meanBuf.resize(voxelNumber);
    sdevBuf.resize(voxelNumber);
    warpedMeanBuf.resize(voxelNumber);
    warpedSdevBuf.resize(voxelNumber);
    correlationBuf.resize(voxelNumber);

    // Upload full per-voxel reference mask to GPU
    refMaskFullCuda.assign(refMask, refMask + voxelNumber);

    if (this->isSymmetric) {
        const size_t voxelNumberBw = NiftiImage::calcVoxelNumber(floImg, 3);
        meanBufBw.resize(voxelNumberBw);
        sdevBufBw.resize(voxelNumberBw);
        warpedMeanBufBw.resize(voxelNumberBw);
        warpedSdevBufBw.resize(voxelNumberBw);
        correlationBufBw.resize(voxelNumberBw);
        floMaskFullCuda.assign(floMask, floMask + voxelNumberBw);
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
double reg_lncc_gpu::GetSimilarityMeasureValueFw() {
    double lncc = 0.0;
    const int *maskPtr = refMaskFullCuda.data().get();
    for (int t = 0; t < this->referenceTimePoints; ++t) {
        if (this->timePointWeights[t] <= 0) continue;
        UpdateLocalStatImagesCuda(this->referenceImage,
                                  this->referenceImageCuda,
                                  this->warpedImageCuda,
                                  maskPtr,
                                  meanBuf.data().get(),
                                  sdevBuf.data().get(),
                                  warpedMeanBuf.data().get(),
                                  warpedSdevBuf.data().get(),
                                  this->kernelStandardDeviation[t],
                                  this->kernelType, t);
        const double tp = GetLnccValueCuda(this->referenceImage,
                                           this->referenceImageCuda,
                                           this->warpedImageCuda,
                                           maskPtr,
                                           meanBuf.data().get(),
                                           sdevBuf.data().get(),
                                           warpedMeanBuf.data().get(),
                                           warpedSdevBuf.data().get(),
                                           correlationBuf.data().get(),
                                           this->kernelStandardDeviation[t],
                                           this->kernelType, t);
        lncc += tp * this->timePointWeights[t];
    }
    return lncc;
}
/* *************************************************************** */
double reg_lncc_gpu::GetSimilarityMeasureValueBw() {
    double lncc = 0.0;
    const int *maskPtr = floMaskFullCuda.data().get();
    for (int t = 0; t < this->referenceTimePoints; ++t) {
        if (this->timePointWeights[t] <= 0) continue;
        UpdateLocalStatImagesCuda(this->floatingImage,
                                  this->floatingImageCuda,
                                  this->warpedImageBwCuda,
                                  maskPtr,
                                  meanBufBw.data().get(),
                                  sdevBufBw.data().get(),
                                  warpedMeanBufBw.data().get(),
                                  warpedSdevBufBw.data().get(),
                                  this->kernelStandardDeviation[t],
                                  this->kernelType, t);
        const double tp = GetLnccValueCuda(this->floatingImage,
                                           this->floatingImageCuda,
                                           this->warpedImageBwCuda,
                                           maskPtr,
                                           meanBufBw.data().get(),
                                           sdevBufBw.data().get(),
                                           warpedMeanBufBw.data().get(),
                                           warpedSdevBufBw.data().get(),
                                           correlationBufBw.data().get(),
                                           this->kernelStandardDeviation[t],
                                           this->kernelType, t);
        lncc += tp * this->timePointWeights[t];
    }
    return lncc;
}
/* *************************************************************** */
void reg_lncc_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    UpdateLocalStatImagesCuda(this->referenceImage,
                              this->referenceImageCuda,
                              this->warpedImageCuda,
                              refMaskFullCuda.data().get(),
                              meanBuf.data().get(),
                              sdevBuf.data().get(),
                              warpedMeanBuf.data().get(),
                              warpedSdevBuf.data().get(),
                              this->kernelStandardDeviation[currentTimePoint],
                              this->kernelType, currentTimePoint);
    GetVoxelBasedLnccGradientCuda(this->referenceImage,
                                  this->referenceImageCuda,
                                  this->warpedImageCuda,
                                  refMaskFullCuda.data().get(),
                                  meanBuf.data().get(),
                                  sdevBuf.data().get(),
                                  warpedMeanBuf.data().get(),
                                  warpedSdevBuf.data().get(),
                                  correlationBuf.data().get(),
                                  this->warpedGradientCuda,
                                  this->voxelBasedGradientCuda,
                                  this->kernelStandardDeviation[currentTimePoint],
                                  this->kernelType, currentTimePoint,
                                  this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
void reg_lncc_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    UpdateLocalStatImagesCuda(this->floatingImage,
                              this->floatingImageCuda,
                              this->warpedImageBwCuda,
                              floMaskFullCuda.data().get(),
                              meanBufBw.data().get(),
                              sdevBufBw.data().get(),
                              warpedMeanBufBw.data().get(),
                              warpedSdevBufBw.data().get(),
                              this->kernelStandardDeviation[currentTimePoint],
                              this->kernelType, currentTimePoint);
    GetVoxelBasedLnccGradientCuda(this->floatingImage,
                                  this->floatingImageCuda,
                                  this->warpedImageBwCuda,
                                  floMaskFullCuda.data().get(),
                                  meanBufBw.data().get(),
                                  sdevBufBw.data().get(),
                                  warpedMeanBufBw.data().get(),
                                  warpedSdevBufBw.data().get(),
                                  correlationBufBw.data().get(),
                                  this->warpedGradientBwCuda,
                                  this->voxelBasedGradientBwCuda,
                                  this->kernelStandardDeviation[currentTimePoint],
                                  this->kernelType, currentTimePoint,
                                  this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
