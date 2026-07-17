/*
 * @file _reg_lncc_gpu.cu
 * @author Marc Modat
 * @date 10/11/2012
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_lncc_gpu.h"
#include "CudaKernelConvolution.hpp"

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
                                     nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda,
                                     size_t activeVoxNumBw) {
    // The CPU initialisation allocates the nifti scratch headers (mean/sdev/correlation and
    // their Bw counterparts) and rescales the reference/floating intensities to [0, 1]
    reg_lncc::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                                localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda, floMask, floMaskCuda, warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda, activeVoxNumBw);

    // reg_lncc::InitialiseMeasure rescaled the reference and floating intensities on the host,
    // so the device copies must be refreshed (mirrors reg_ssd_gpu for normalised time points).
    // The floating image must be up to date before the warped image is (re)computed on the device.
    Cuda::TransferNiftiToDevice(this->referenceImageCuda, this->referenceImage);
    Cuda::TransferNiftiToDevice(this->floatingImageCuda, this->floatingImage);

    // Allocate the device scratch buffers (see _reg_lncc_gpu.h for the lane layout) and upload
    // the per-voxel host mask once - the per-call combined-mask build is then a pure device pass
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);
    this->statsImageCuda.resize(voxelNumber);
    this->correlationImageCuda.resize(voxelNumber);
    this->forwardMaskCuda.resize(voxelNumber);
    this->baseMaskCuda.resize(voxelNumber);
    thrust::copy_n(this->referenceMask, voxelNumber, this->baseMaskCuda.begin());
    if (this->isSymmetric) {
        const size_t voxelNumberBw = NiftiImage::calcVoxelNumber(this->floatingImage, 3);
        this->statsImageBwCuda.resize(voxelNumberBw);
        this->correlationImageBwCuda.resize(voxelNumberBw);
        this->backwardMaskCuda.resize(voxelNumberBw);
        this->baseMaskBwCuda.resize(voxelNumberBw);
        thrust::copy_n(this->floatingMask, voxelNumberBw, this->baseMaskBwCuda.begin());
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
namespace {
/* *************************************************************** */
// Runtime dispatch to the compile-time templated packed convolution (all four float4 lanes in one
// pass). LNCC accumulates in float (not double)
void ConvolvePacked(const nifti_image *image, float4 *imageCuda, const float *sigma,
                    const ConvKernelType kernelType, const int *maskCuda,
                    NiftyReg::Cuda::KernelConvolutionWorkspace *workspace) {
    switch (kernelType) {
    case ConvKernelType::Mean:
        NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Mean, float>(image, imageCuda, sigma, maskCuda, workspace);
        break;
    case ConvKernelType::Linear:
        NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Linear, float>(image, imageCuda, sigma, maskCuda, workspace);
        break;
    case ConvKernelType::Gaussian:
        NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Gaussian, float>(image, imageCuda, sigma, maskCuda, workspace);
        break;
    case ConvKernelType::Cubic:
        NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Cubic, float>(image, imageCuda, sigma, maskCuda, workspace);
        break;
    default:
        NR_FATAL_ERROR("Unsupported kernel type");
    }
}
/* *************************************************************** */
// Single-channel (.x) dispatch, used for the correlation image: a packed pass would read 16 bytes
// per tap for one useful lane, whose larger cache footprint hurts at large volume sizes, whereas
// the single-channel axis kernels work on compact float scratch (4 bytes per tap).
void Convolve(const nifti_image *image, float4 *imageCuda, const float *sigma,
              const ConvKernelType kernelType, const int *maskCuda,
              NiftyReg::Cuda::KernelConvolutionWorkspace *workspace) {
    switch (kernelType) {
    case ConvKernelType::Mean:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Mean, float>(image, imageCuda, sigma, nullptr, nullptr, maskCuda, workspace);
        break;
    case ConvKernelType::Linear:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Linear, float>(image, imageCuda, sigma, nullptr, nullptr, maskCuda, workspace);
        break;
    case ConvKernelType::Gaussian:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Gaussian, float>(image, imageCuda, sigma, nullptr, nullptr, maskCuda, workspace);
        break;
    case ConvKernelType::Cubic:
        NiftyReg::Cuda::KernelConvolution<ConvKernelType::Cubic, float>(image, imageCuda, sigma, nullptr, nullptr, maskCuda, workspace);
        break;
    default:
        NR_FATAL_ERROR("Unsupported kernel type");
    }
}
/* *************************************************************** */
// Build the per-voxel combined mask on the device in a single pass: start from the device-resident
// base mask (reg_measure convention: value >= 0 when active, -1 otherwise) and remove every voxel
// that is NaN in the reference or warped image at any time point, mirroring the CPU
// memcpy(combinedMask, refMask) + reg_tools_removeNanFromMask sequence in UpdateLocalStatImages.
void BuildCombinedMask(const int *baseMaskCuda, int *combinedMaskCuda, const size_t voxelNumber,
                       const float *referenceImageCuda, const float *warpedImageCuda,
                       const int referenceTimePoints) {
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
                       [=]__device__(const size_t index) {
        int value = baseMaskCuda[index];
        for (int t = 0; t < referenceTimePoints; ++t) {
            const float refVal = referenceImageCuda[t * voxelNumber + index];
            const float warVal = warpedImageCuda[t * voxelNumber + index];
            if (refVal != refVal || warVal != warVal) {
                value = -1;
                break;
            }
        }
        combinedMaskCuda[index] = value;
    });
}
/* *************************************************************** */
// Port of UpdateLocalStatImages (_reg_lncc.cpp). Fills the packed stats image with the raw
// (pre-finalisation) smoothed statistics for the given time point: x = G*I, y = G*(I^2),
// z = G*J, w = G*(J^2). Per lane the convolution is bit-identical to smoothing each CPU image.
void UpdateLocalStatImages_gpu(const nifti_image *image,
                               const float *referenceImageCuda,
                               const float *warpedImageCuda,
                               float4 *statsImageCuda,
                               const int *combinedMaskCuda,
                               const float *kernelStandardDeviation,
                               const ConvKernelType kernelType,
                               const int currentTimePoint,
                               NiftyReg::Cuda::KernelConvolutionWorkspace& workspace) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const float *refPtr = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warPtr = warpedImageCuda + currentTimePoint * voxelNumber;

    // The combined mask (hence the smoothed density) is the same for every convolution of this
    // call, so the packed convolution computes the density once and the correlation/gradient
    // convolutions that follow reuse it.
    workspace.densityValid = false;

    // stats = (I, I^2, J, J^2)
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
                       [=]__device__(const size_t index) {
        const float refVal = refPtr[index];
        const float warVal = warPtr[index];
        statsImageCuda[index] = make_float4(refVal, refVal * refVal, warVal, warVal * warVal);
    });

    ConvolvePacked(image, statsImageCuda, kernelStandardDeviation, kernelType, combinedMaskCuda, &workspace);
}
/* *************************************************************** */
// Finalise the smoothed statistics (sdev = sqrt(G*(I^2) - (G*I)^2), stabilised - identical float
// expressions to the CPU) and, in the same pass, initialise the correlation image with ref * war.
void FinaliseStatsInitCorrelation(const size_t voxelNumber,
                                  float4 *statsImageCuda,
                                  float4 *correlationImageCuda,
                                  const float *refPtr,
                                  const float *warPtr) {
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
                       [=]__device__(const size_t index) {
        const float4 stats = statsImageCuda[index];
        float sdev = sqrtf(stats.y - stats.x * stats.x);
        if (sdev < 1.e-06) sdev = 0;
        float warSdev = sqrtf(stats.w - stats.z * stats.z);
        if (warSdev < 1.e-06) warSdev = 0;
        statsImageCuda[index] = make_float4(stats.x, sdev, stats.z, warSdev);
        correlationImageCuda[index] = make_float4(refPtr[index] * warPtr[index], 0, 0, 0);
    });
}
/* *************************************************************** */
// Port of reg_getLnccValue (_reg_lncc.cpp) for a single time point.
double GetLnccValue_gpu(const nifti_image *image,
                        const float *referenceImageCuda,
                        const float *warpedImageCuda,
                        float4 *statsImageCuda,
                        float4 *correlationImageCuda,
                        const int *combinedMaskCuda,
                        const float *kernelStandardDeviation,
                        const ConvKernelType kernelType,
                        const int currentTimePoint,
                        NiftyReg::Cuda::KernelConvolutionWorkspace& workspace) {
    UpdateLocalStatImages_gpu(image, referenceImageCuda, warpedImageCuda, statsImageCuda,
                              combinedMaskCuda, kernelStandardDeviation, kernelType, currentTimePoint, workspace);

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const float *refPtr = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warPtr = warpedImageCuda + currentTimePoint * voxelNumber;

    FinaliseStatsInitCorrelation(voxelNumber, statsImageCuda, correlationImageCuda, refPtr, warPtr);
    // correlation = G*(ref * war); reuses the density cached by the stats convolution
    Convolve(image, correlationImageCuda, kernelStandardDeviation, kernelType, combinedMaskCuda, &workspace);

    // Sum |lncc| over the active voxels where the value is finite; also count them
    const double2 lnccSumAndCount = thrust::transform_reduce(thrust::device,
        thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(voxelNumber),
        [=]__device__(const size_t index) -> double2 {
            if (combinedMaskCuda[index] > -1) {
                // Mirror the CPU float arithmetic exactly before widening to double
                const float4 stats = statsImageCuda[index];
                const float num = correlationImageCuda[index].x - stats.x * stats.z;
                const float den = stats.y * stats.w;
                const float lncc = num / den;
                if (lncc == lncc && !isinf(lncc))
                    return make_double2(fabs(static_cast<double>(lncc)), 1.0);
            }
            return make_double2(0, 0);
        }, make_double2(0, 0), thrust::plus<double2>());

    return lnccSumAndCount.x / lnccSumAndCount.y;
}
/* *************************************************************** */
// Port of reg_getVoxelBasedLnccGradient (_reg_lncc.cpp) for a single time point.
void GetVoxelBasedLnccGradient_gpu(const nifti_image *image,
                                   const bool is3d,
                                   const float *referenceImageCuda,
                                   const float *warpedImageCuda,
                                   float4 *statsImageCuda,
                                   float4 *correlationImageCuda,
                                   const float4 *warpedGradientCuda,
                                   float4 *voxelBasedGradientCuda,
                                   const int *combinedMaskCuda,
                                   const float *kernelStandardDeviation,
                                   const ConvKernelType kernelType,
                                   const int currentTimePoint,
                                   const double timePointWeight,
                                   NiftyReg::Cuda::KernelConvolutionWorkspace& workspace) {
    UpdateLocalStatImages_gpu(image, referenceImageCuda, warpedImageCuda, statsImageCuda,
                              combinedMaskCuda, kernelStandardDeviation, kernelType, currentTimePoint, workspace);

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const float *refPtr = referenceImageCuda + currentTimePoint * voxelNumber;
    const float *warPtr = warpedImageCuda + currentTimePoint * voxelNumber;

    FinaliseStatsInitCorrelation(voxelNumber, statsImageCuda, correlationImageCuda, refPtr, warPtr);
    // correlation = G*(ref * war); reuses the density cached by the stats convolution
    Convolve(image, correlationImageCuda, kernelStandardDeviation, kernelType, combinedMaskCuda, &workspace);

    // Compute temp1/temp2/temp3 (identical double expressions to the CPU), store them into the
    // correlation image lanes x/y/z for the subsequent packed smoothing, and count the voxels
    // that contribute a finite gradient term - a single pass replacing the previous
    // count_if + store passes (matches the CPU activeVoxelNumber exactly: an integer count is
    // reduction-order independent).
    const size_t activeVoxelNumber = thrust::transform_reduce(thrust::device,
        thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(voxelNumber),
        [=]__device__(const size_t index) -> size_t {
            if (combinedMaskCuda[index] > -1) {
                const float4 stats = statsImageCuda[index];
                const double refMeanValue = stats.x;
                const double warMeanValue = stats.z;
                const double refSdevValue = stats.y;
                const double warSdevValue = stats.w;
                const double correlaValue = static_cast<double>(correlationImageCuda[index].x) - (refMeanValue * warMeanValue);
                double temp1 = 1.0 / (refSdevValue * warSdevValue);
                double temp2 = correlaValue / (refSdevValue * warSdevValue * warSdevValue * warSdevValue);
                double temp3 = (correlaValue * warMeanValue) / (refSdevValue * warSdevValue * warSdevValue * warSdevValue)
                    - refMeanValue / (refSdevValue * warSdevValue);
                if (temp1 == temp1 && !isinf(temp1) &&
                    temp2 == temp2 && !isinf(temp2) &&
                    temp3 == temp3 && !isinf(temp3)) {
                    // Derivative of the absolute function
                    if (correlaValue < 0) {
                        temp1 *= -1;
                        temp2 *= -1;
                        temp3 *= -1;
                    }
                    correlationImageCuda[index] = make_float4(static_cast<float>(temp1),
                                                              static_cast<float>(temp2),
                                                              static_cast<float>(temp3), 0);
                    return 1;
                }
            }
            correlationImageCuda[index] = make_float4(0, 0, 0, 0);
            return 0;
        }, size_t(0), thrust::plus<size_t>());

    const double adjustedWeight = timePointWeight / activeVoxelNumber;

    // Smooth the three temp fields in one packed pass (same combined mask, so the cached density
    // is reused; lane w carries zeros)
    ConvolvePacked(image, correlationImageCuda, kernelStandardDeviation, kernelType, combinedMaskCuda, &workspace);

    // Accumulate the gradient
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
                       [=]__device__(const size_t index) {
        if (combinedMaskCuda[index] > -1) {
            const float4 temps = correlationImageCuda[index];
            const float inner = temps.x * refPtr[index]
                - temps.y * warPtr[index]
                + temps.z;
            const double common = static_cast<double>(inner) * adjustedWeight;
            const float4 warpGradient = warpedGradientCuda[index];
            float4 measureGradient = voxelBasedGradientCuda[index];
            measureGradient.x -= static_cast<float>(warpGradient.x * common);
            measureGradient.y -= static_cast<float>(warpGradient.y * common);
            if (is3d)
                measureGradient.z -= static_cast<float>(warpGradient.z * common);
            voxelBasedGradientCuda[index] = measureGradient;
        }
    });

    // Zero out any NaN/Inf produced in the gradient
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber,
                       [=]__device__(const size_t index) {
        float4 measureGradient = voxelBasedGradientCuda[index];
        if (measureGradient.x != measureGradient.x || isinf(measureGradient.x)) measureGradient.x = 0;
        if (measureGradient.y != measureGradient.y || isinf(measureGradient.y)) measureGradient.y = 0;
        if (is3d && (measureGradient.z != measureGradient.z || isinf(measureGradient.z))) measureGradient.z = 0;
        voxelBasedGradientCuda[index] = measureGradient;
    });
}
/* *************************************************************** */
} // anonymous namespace
/* *************************************************************** */
double reg_lncc_gpu::GetSimilarityMeasureValueFw() {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);
    int *combinedMaskCuda = this->forwardMaskCuda.data().get();
    BuildCombinedMask(this->baseMaskCuda.data().get(), combinedMaskCuda, voxelNumber,
                      this->referenceImageCuda, this->warpedImageCuda, this->referenceTimePoints);

    double lncc = 0;
    for (int currentTimePoint = 0; currentTimePoint < this->referenceTimePoints; ++currentTimePoint) {
        if (this->timePointWeights[currentTimePoint] > 0) {
            const double tp = GetLnccValue_gpu(this->correlationImage,
                                               this->referenceImageCuda,
                                               this->warpedImageCuda,
                                               this->statsImageCuda.data().get(),
                                               this->correlationImageCuda.data().get(),
                                               combinedMaskCuda,
                                               this->kernelStandardDeviation,
                                               this->kernelType,
                                               currentTimePoint,
                                               this->convWorkspace);
            lncc += tp * this->timePointWeights[currentTimePoint];
        }
    }
    return lncc;
}
/* *************************************************************** */
double reg_lncc_gpu::GetSimilarityMeasureValueBw() {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->floatingImage, 3);
    int *combinedMaskCuda = this->backwardMaskCuda.data().get();
    BuildCombinedMask(this->baseMaskBwCuda.data().get(), combinedMaskCuda, voxelNumber,
                      this->floatingImageCuda, this->warpedImageBwCuda, this->referenceTimePoints);

    double lncc = 0;
    for (int currentTimePoint = 0; currentTimePoint < this->referenceTimePoints; ++currentTimePoint) {
        if (this->timePointWeights[currentTimePoint] > 0) {
            const double tp = GetLnccValue_gpu(this->correlationImageBw,
                                               this->floatingImageCuda,
                                               this->warpedImageBwCuda,
                                               this->statsImageBwCuda.data().get(),
                                               this->correlationImageBwCuda.data().get(),
                                               combinedMaskCuda,
                                               this->kernelStandardDeviation,
                                               this->kernelType,
                                               currentTimePoint,
                                               this->convWorkspace);
            lncc += tp * this->timePointWeights[currentTimePoint];
        }
    }
    return lncc;
}
/* *************************************************************** */
void reg_lncc_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);
    int *combinedMaskCuda = this->forwardMaskCuda.data().get();
    BuildCombinedMask(this->baseMaskCuda.data().get(), combinedMaskCuda, voxelNumber,
                      this->referenceImageCuda, this->warpedImageCuda, this->referenceTimePoints);
    GetVoxelBasedLnccGradient_gpu(this->correlationImage,
                                  this->referenceImage->nz > 1,
                                  this->referenceImageCuda,
                                  this->warpedImageCuda,
                                  this->statsImageCuda.data().get(),
                                  this->correlationImageCuda.data().get(),
                                  this->warpedGradientCuda,
                                  this->voxelBasedGradientCuda,
                                  combinedMaskCuda,
                                  this->kernelStandardDeviation,
                                  this->kernelType,
                                  currentTimePoint,
                                  this->timePointWeights[currentTimePoint],
                                  this->convWorkspace);
}
/* *************************************************************** */
void reg_lncc_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(this->floatingImage, 3);
    int *combinedMaskCuda = this->backwardMaskCuda.data().get();
    BuildCombinedMask(this->baseMaskBwCuda.data().get(), combinedMaskCuda, voxelNumber,
                      this->floatingImageCuda, this->warpedImageBwCuda, this->referenceTimePoints);
    GetVoxelBasedLnccGradient_gpu(this->correlationImageBw,
                                  this->floatingImage->nz > 1,
                                  this->floatingImageCuda,
                                  this->warpedImageBwCuda,
                                  this->statsImageBwCuda.data().get(),
                                  this->correlationImageBwCuda.data().get(),
                                  this->warpedGradientBwCuda,
                                  this->voxelBasedGradientBwCuda,
                                  combinedMaskCuda,
                                  this->kernelStandardDeviation,
                                  this->kernelType,
                                  currentTimePoint,
                                  this->timePointWeights[currentTimePoint],
                                  this->convWorkspace);
}
/* *************************************************************** */
