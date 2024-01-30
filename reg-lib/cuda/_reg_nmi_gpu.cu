/**
 * @file _reg_nmi_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_nmi_gpu.h"
#include "_reg_common_cuda_kernels.cu"

/* *************************************************************** */
reg_nmi_gpu::reg_nmi_gpu(): reg_nmi::reg_nmi() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_nmi_gpu::~reg_nmi_gpu() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_nmi_gpu::InitialiseMeasure(nifti_image *refImg, float *refImgCuda,
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
    reg_nmi::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda, floMask, floMaskCuda, warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda);
    // The reference and floating images have to be updated on the device
    Cuda::TransferNiftiToDevice(this->referenceImageCuda, this->referenceImage);
    Cuda::TransferNiftiToDevice(this->floatingImageCuda, this->floatingImage);
    // Create the joint histograms
    this->jointHistogramLogCudaVecs.resize(this->referenceTimePoints);
    this->jointHistogramProCudaVecs.resize(this->referenceTimePoints);
    if (this->isSymmetric) {
        this->jointHistogramLogBwCudaVecs.resize(this->referenceTimePoints);
        this->jointHistogramProBwCudaVecs.resize(this->referenceTimePoints);
    }
    for (int i = 0; i < this->referenceTimePoints; i++) {
        if (this->timePointWeights[i] > 0) {
            this->jointHistogramLogCudaVecs[i].resize(this->totalBinNumber[i]);
            this->jointHistogramProCudaVecs[i].resize(this->totalBinNumber[i]);
            if (this->isSymmetric) {
                this->jointHistogramLogBwCudaVecs[i].resize(this->totalBinNumber[i]);
                this->jointHistogramProBwCudaVecs[i].resize(this->totalBinNumber[i]);
            }
        }
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_getNmiValue_gpu(const nifti_image *referenceImage,
                         const float *referenceImageCuda,
                         const float *warpedImageCuda,
                         const double *timePointWeights,
                         const int referenceTimePoints,
                         const unsigned short *referenceBinNumber,
                         const unsigned short *floatingBinNumber,
                         const unsigned short *totalBinNumber,
                         vector<thrust::device_vector<double>>& jointHistogramLogCudaVecs,
                         vector<thrust::device_vector<double>>& jointHistogramProCudaVecs,
                         double **entropyValues,
                         const int *maskCuda,
                         const size_t activeVoxelNumber,
                         const bool approximation) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const int3 referenceImageDims = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);

    // Iterate over all active time points
    for (int t = 0; t < referenceTimePoints; t++) {
        if (timePointWeights[t] <= 0) continue;
        NR_DEBUG("Computing NMI for time point " << t);
        const auto curTotalBinNumber = totalBinNumber[t];
        const auto curRefBinNumber = referenceBinNumber[t];
        const auto curFloBinNumber = floatingBinNumber[t];
        // Define the current histograms
        thrust::fill(thrust::device, jointHistogramLogCudaVecs[t].begin(), jointHistogramLogCudaVecs[t].end(), 0.0);
        thrust::fill(thrust::device, jointHistogramProCudaVecs[t].begin(), jointHistogramProCudaVecs[t].end(), 0.0);
        double *jointHistogramLogCuda = jointHistogramLogCudaVecs[t].data().get();
        double *jointHistogramProCuda = jointHistogramProCudaVecs[t].data().get();
        // Define the current textures
        auto referenceImageTexturePtr = Cuda::CreateTextureObject(referenceImageCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        auto warpedImageTexturePtr = Cuda::CreateTextureObject(warpedImageCuda + t * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
        auto referenceImageTexture = *referenceImageTexturePtr;
        auto warpedImageTexture = *warpedImageTexturePtr;
        // Fill the joint histograms
        if (approximation == false) {
            // No approximation is used for the Parzen windowing
            thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
                const float refValue = tex1Dfetch<float>(referenceImageTexture, index);
                if (refValue != refValue) return;
                const float warValue = tex1Dfetch<float>(warpedImageTexture, index);
                if (warValue != warValue) return;
                for (int r = int(refValue) - 1; r < int(refValue) + 3; r++) {
                    if (0 <= r && r < curRefBinNumber) {
                        const double refBasis = GetBasisSplineValue<double>(refValue - r);
                        for (int w = int(warValue) - 1; w < int(warValue) + 3; w++) {
                            if (0 <= w && w < curFloBinNumber) {
                                const double warBasis = GetBasisSplineValue<double>(warValue - w);
                                atomicAdd(&jointHistogramProCuda[r + w * curRefBinNumber], refBasis * warBasis);
                            }
                        }
                    }
                }
            });
        } else {
            // An approximation is used for the Parzen windowing. First intensities are binarised then
            // the histogram is convolved with a spine kernel function.
            thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
                const float refValue = tex1Dfetch<float>(referenceImageTexture, index);
                if (refValue != refValue) return;
                const float warValue = tex1Dfetch<float>(warpedImageTexture, index);
                if (warValue != warValue) return;
                if (0 <= refValue && refValue < curRefBinNumber && 0 <= warValue && warValue < curFloBinNumber)
                    atomicAdd(&jointHistogramProCuda[int(refValue) + int(warValue) * curRefBinNumber], 1.0);
            });
            // Convolve the histogram with a cubic B-spline kernel
            // Histogram is first smooth along the reference axis
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned short>(0), curFloBinNumber, [=]__device__(const unsigned short f) {
                constexpr double kernel[3]{ GetBasisSplineValue(-1.0), GetBasisSplineValue(0.0), GetBasisSplineValue(-1.0) };
                for (unsigned short r = 0; r < curRefBinNumber; r++) {
                    double value = 0;
                    short index = r - 1;
                    double *histoPtr = &jointHistogramProCuda[index + curRefBinNumber * f];

                    for (char it = 0; it < 3; it++, index++, histoPtr++)
                        if (-1 < index && index < curRefBinNumber)
                            value += *histoPtr * kernel[it];
                    jointHistogramLogCuda[r + curRefBinNumber * f] = value;
                }
            });
            // Histogram is then smooth along the warped floating axis
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned short>(0), curRefBinNumber, [=]__device__(const unsigned short r) {
                constexpr double kernel[3]{ GetBasisSplineValue(-1.0), GetBasisSplineValue(0.0), GetBasisSplineValue(-1.0) };
                for (unsigned short f = 0; f < curFloBinNumber; f++) {
                    double value = 0;
                    short index = f - 1;
                    double *histoPtr = &jointHistogramLogCuda[r + curRefBinNumber * index];

                    for (char it = 0; it < 3; it++, index++, histoPtr += curRefBinNumber)
                        if (-1 < index && index < curFloBinNumber)
                            value += *histoPtr * kernel[it];
                    jointHistogramProCuda[r + curRefBinNumber * f] = value;
                }
            });
        }
        // Normalise the histogram
        const double activeVoxel = thrust::reduce(thrust::device, jointHistogramProCudaVecs[t].begin(), jointHistogramProCudaVecs[t].end(), 0.0, thrust::plus<double>());
        entropyValues[t][3] = activeVoxel;
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned>(0), curTotalBinNumber, [=]__device__(const unsigned index) {
            jointHistogramProCuda[index] /= activeVoxel;
        });
        // Marginalise over the reference axis
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned short>(0), curRefBinNumber, [=]__device__(const unsigned short r) {
            double sum = 0;
            unsigned short index = r;
            for (unsigned short f = 0; f < curFloBinNumber; f++, index += curRefBinNumber)
                sum += jointHistogramProCuda[index];
            jointHistogramProCuda[curRefBinNumber * curFloBinNumber + r] = sum;
        });
        // Marginalise over the warped floating axis
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned short>(0), curFloBinNumber, [=]__device__(const unsigned short f) {
            double sum = 0;
            unsigned short index = curRefBinNumber * f;
            for (unsigned short r = 0; r < curRefBinNumber; r++, index++)
                sum += jointHistogramProCuda[index];
            jointHistogramProCuda[curRefBinNumber * curFloBinNumber + curRefBinNumber + f] = sum;
        });
        // Compute the entropy of the reference image
        thrust::counting_iterator<unsigned short> it(0);
        entropyValues[t][0] = thrust::transform_reduce(thrust::device, it, it + curRefBinNumber, [=]__device__(const unsigned short r) {
            const double valPro = jointHistogramProCuda[curRefBinNumber * curFloBinNumber + r];
            if (valPro > 0) {
                const double valLog = log(valPro);
                jointHistogramLogCuda[curRefBinNumber * curFloBinNumber + r] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
        // Compute the entropy of the warped floating image
        it = thrust::counting_iterator<unsigned short>(0);
        entropyValues[t][1] = thrust::transform_reduce(thrust::device, it, it + curFloBinNumber, [=]__device__(const unsigned short f) {
            const double valPro = jointHistogramProCuda[curRefBinNumber * curFloBinNumber + curRefBinNumber + f];
            if (valPro > 0) {
                const double valLog = log(valPro);
                jointHistogramLogCuda[curRefBinNumber * curFloBinNumber + curRefBinNumber + f] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
        // Compute the joint entropy
        it = thrust::counting_iterator<unsigned short>(0);
        entropyValues[t][2] = thrust::transform_reduce(thrust::device, it, it + curRefBinNumber * curFloBinNumber, [=]__device__(const unsigned short index) {
            const double valPro = jointHistogramProCuda[index];
            if (valPro > 0) {
                const double valLog = log(valPro);
                jointHistogramLogCuda[index] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
    } // iterate over all time point in the reference image
}
/* *************************************************************** */
static double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                        const float *referenceImageCuda,
                                        const nifti_image *warpedImage,
                                        const float *warpedImageCuda,
                                        const double *timePointWeights,
                                        const int referenceTimePoints,
                                        const unsigned short *referenceBinNumber,
                                        const unsigned short *floatingBinNumber,
                                        const unsigned short *totalBinNumber,
                                        vector<thrust::device_vector<double>>& jointHistogramLogCudaVecs,
                                        vector<thrust::device_vector<double>>& jointHistogramProCudaVecs,
                                        double **entropyValues,
                                        const int *referenceMaskCuda,
                                        const size_t activeVoxelNumber,
                                        const bool approximation) {
    reg_getNmiValue_gpu(referenceImage,
                        referenceImageCuda,
                        warpedImageCuda,
                        timePointWeights,
                        referenceTimePoints,
                        referenceBinNumber,
                        floatingBinNumber,
                        totalBinNumber,
                        jointHistogramLogCudaVecs,
                        jointHistogramProCudaVecs,
                        entropyValues,
                        referenceMaskCuda,
                        activeVoxelNumber,
                        approximation);

    double nmi = 0;
    for (int t = 0; t < referenceTimePoints; t++) {
        if (timePointWeights[t] > 0)
            nmi += timePointWeights[t] * (entropyValues[t][0] + entropyValues[t][1]) / entropyValues[t][2];
    }
    return nmi;
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->referenceImageCuda,
                                       this->warpedImage,
                                       this->warpedImageCuda,
                                       this->timePointWeights,
                                       this->referenceTimePoints,
                                       this->referenceBinNumber,
                                       this->floatingBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLogCudaVecs,
                                       this->jointHistogramProCudaVecs,
                                       this->entropyValues,
                                       this->referenceMaskCuda,
                                       this->activeVoxelNumber,
                                       this->approximatePw);
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->floatingImageCuda,
                                       this->warpedImageBw,
                                       this->warpedImageBwCuda,
                                       this->timePointWeights,
                                       this->referenceTimePoints,
                                       this->floatingBinNumber,
                                       this->referenceBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLogBwCudaVecs,
                                       this->jointHistogramProBwCudaVecs,
                                       this->entropyValuesBw,
                                       this->floatingMaskCuda,
                                       this->activeVoxelNumber,
                                       this->approximatePw);
}
/* *************************************************************** */
template<bool is3d> struct Derivative { using Type = double3; };
template<> struct Derivative<false> { using Type = double2; };
/* *************************************************************** */
/// Called when we only have one target and one source image
template<bool is3d>
void reg_getVoxelBasedNmiGradient_gpu(const nifti_image *referenceImage,
                                      const float *referenceImageCuda,
                                      const float *warpedImageCuda,
                                      const float4 *warpedGradientCuda,
                                      const double *jointHistogramLogCuda,
                                      float4 *voxelBasedGradientCuda,
                                      const int *maskCuda,
                                      const size_t activeVoxelNumber,
                                      const double *entropies,
                                      const int refBinNumber,
                                      const int floBinNumber,
                                      const int totalBinNumber,
                                      const double timePointWeight,
                                      const int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const int3 imageSize = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const double normalisedJE = entropies[2] * entropies[3];
    const double nmi = (entropies[0] + entropies[1]) / entropies[2];
    const int referenceOffset = refBinNumber * floBinNumber;
    const int floatingOffset = referenceOffset + refBinNumber;

    auto referenceImageTexturePtr = Cuda::CreateTextureObject(referenceImageCuda + currentTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto warpedImageTexturePtr = Cuda::CreateTextureObject(warpedImageCuda + currentTimePoint * voxelNumber, voxelNumber, cudaChannelFormatKindFloat, 1);
    auto warpedGradientTexturePtr = Cuda::CreateTextureObject(warpedGradientCuda, voxelNumber, cudaChannelFormatKindFloat, 4);
    auto referenceImageTexture = *referenceImageTexturePtr;
    auto warpedImageTexture = *warpedImageTexturePtr;
    auto warpedGradientTexture = *warpedGradientTexturePtr;

    thrust::for_each_n(thrust::device, maskCuda, activeVoxelNumber, [=]__device__(const int index) {
        const float refValue = tex1Dfetch<float>(referenceImageTexture, index);
        if (refValue != refValue) return;
        const float warValue = tex1Dfetch<float>(warpedImageTexture, index);
        if (warValue != warValue) return;
        const float4 warGradValue = tex1Dfetch<float4>(warpedGradientTexture, index);

        // No computation is performed if any of the point is part of the background
        // The two is added because the image is resample between 2 and bin+2
        // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65
        typename Derivative<is3d>::Type jointDeriv{}, refDeriv{}, warDeriv{};
        for (int r = int(refValue) - 1; r < int(refValue) + 3; r++) {
            if (-1 < r && r < refBinNumber) {
                for (int w = int(warValue) - 1; w < int(warValue) + 3; w++) {
                    if (-1 < w && w < floBinNumber) {
                        const double commonValue = (GetBasisSplineValue<double>(refValue - r) *
                                                    GetBasisSplineDerivativeValue<double>(warValue - w));
                        const double jointLog = jointHistogramLogCuda[r + w * refBinNumber];
                        const double refLog = jointHistogramLogCuda[r + referenceOffset];
                        const double warLog = jointHistogramLogCuda[w + floatingOffset];
                        if (warGradValue.x == warGradValue.x) {
                            const double commonMultGrad = commonValue * warGradValue.x;
                            jointDeriv.x += commonMultGrad * jointLog;
                            refDeriv.x += commonMultGrad * refLog;
                            warDeriv.x += commonMultGrad * warLog;
                        }
                        if (warGradValue.y == warGradValue.y) {
                            const double commonMultGrad = commonValue * warGradValue.y;
                            jointDeriv.y += commonMultGrad * jointLog;
                            refDeriv.y += commonMultGrad * refLog;
                            warDeriv.y += commonMultGrad * warLog;
                        }
                        if constexpr (is3d) {
                            if (warGradValue.z == warGradValue.z) {
                                const double commonMultGrad = commonValue * warGradValue.z;
                                jointDeriv.z += commonMultGrad * jointLog;
                                refDeriv.z += commonMultGrad * refLog;
                                warDeriv.z += commonMultGrad * warLog;
                            }
                        }
                    }
                }
            }
        }

        // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way
        float4 gradValue = voxelBasedGradientCuda[index];
        gradValue.x += static_cast<float>(timePointWeight * (refDeriv.x + warDeriv.x - nmi * jointDeriv.x) / normalisedJE);
        gradValue.y += static_cast<float>(timePointWeight * (refDeriv.y + warDeriv.y - nmi * jointDeriv.y) / normalisedJE);
        if constexpr (is3d)
            gradValue.z += static_cast<float>(timePointWeight * (refDeriv.z + warDeriv.z - nmi * jointDeriv.z) / normalisedJE);
        voxelBasedGradientCuda[index] = gradValue;
    });
}
/* *************************************************************** */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    // Call compute similarity measure to calculate joint histogram
    this->GetSimilarityMeasureValue();

    auto getVoxelBasedNmiGradient = this->referenceImage->nz > 1 ? reg_getVoxelBasedNmiGradient_gpu<true> : reg_getVoxelBasedNmiGradient_gpu<false>;
    getVoxelBasedNmiGradient(this->referenceImage,
                             this->referenceImageCuda,
                             this->warpedImageCuda,
                             this->warpedGradientCuda,
                             this->jointHistogramLogCudaVecs[currentTimePoint].data().get(),
                             this->voxelBasedGradientCuda,
                             this->referenceMaskCuda,
                             this->activeVoxelNumber,
                             this->entropyValues[currentTimePoint],
                             this->referenceBinNumber[currentTimePoint],
                             this->floatingBinNumber[currentTimePoint],
                             this->totalBinNumber[currentTimePoint],
                             this->timePointWeights[currentTimePoint],
                             currentTimePoint);
}
/* *************************************************************** */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    auto getVoxelBasedNmiGradient = this->floatingImage->nz > 1 ? reg_getVoxelBasedNmiGradient_gpu<true> : reg_getVoxelBasedNmiGradient_gpu<false>;
    getVoxelBasedNmiGradient(this->floatingImage,
                             this->floatingImageCuda,
                             this->warpedImageBwCuda,
                             this->warpedGradientBwCuda,
                             this->jointHistogramLogBwCudaVecs[currentTimePoint].data().get(),
                             this->voxelBasedGradientBwCuda,
                             this->floatingMaskCuda,
                             this->activeVoxelNumber,
                             this->entropyValuesBw[currentTimePoint],
                             this->floatingBinNumber[currentTimePoint],
                             this->referenceBinNumber[currentTimePoint],
                             this->totalBinNumber[currentTimePoint],
                             this->timePointWeights[currentTimePoint],
                             currentTimePoint);
}
/* *************************************************************** */
