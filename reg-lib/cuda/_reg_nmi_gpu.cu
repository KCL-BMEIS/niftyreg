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
#include "_reg_nmi_kernels.cu"

/* *************************************************************** */
reg_nmi_gpu::reg_nmi_gpu(): reg_nmi::reg_nmi() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_nmi_gpu::~reg_nmi_gpu() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_nmi_gpu::InitialiseMeasure(nifti_image *refImg, cudaArray *refImgCuda,
                                    nifti_image *floImg, cudaArray *floImgCuda,
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
    // Check if the input images have multiple time points
    if (this->referenceTimePoints > 1 || this->floatingImage->nt > 1)
        NR_FATAL_ERROR("Multiple time points are not yet supported");
    // The reference and floating images have to be updated on the device
    Cuda::TransferNiftiToDevice<float>(this->referenceImageCuda, this->referenceImage);
    Cuda::TransferNiftiToDevice<float>(this->floatingImageCuda, this->floatingImage);
    // Create the joint histograms
    this->jointHistogramLogCudaVecs.resize(this->referenceTimePoints);
    this->jointHistogramProCudaVecs.resize(this->referenceTimePoints);
    if (this->isSymmetric) {
        this->jointHistogramLogBwCudaVecs.resize(this->referenceTimePoints);
        this->jointHistogramProBwCudaVecs.resize(this->referenceTimePoints);
    }
    for (int i = 0; i < this->referenceTimePoints; ++i) {
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
                         const cudaArray *referenceImageCuda,
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
    auto referenceImageTexturePtr = Cuda::CreateTextureObject(referenceImageCuda, cudaResourceTypeArray);
    auto maskTexturePtr = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                    cudaChannelFormatKindSigned, 1);
    auto referenceImageTexture = *referenceImageTexturePtr;
    auto maskTexture = *maskTexturePtr;

    // Iterate over all active time points
    for (int t = 0; t < referenceTimePoints; t++) {
        if (timePointWeights[t] <= 0) continue;
        NR_DEBUG("Computing NMI for time point " << t);
        const auto& curTotalBinNumber = totalBinNumber[t];
        const auto& curRefBinNumber = referenceBinNumber[t];
        const auto& curFloBinNumber = floatingBinNumber[t];
        // Define the current histograms
        thrust::fill(thrust::device, jointHistogramLogCudaVecs[t].begin(), jointHistogramLogCudaVecs[t].end(), 0.0);
        thrust::fill(thrust::device, jointHistogramProCudaVecs[t].begin(), jointHistogramProCudaVecs[t].end(), 0.0);
        double *jointHistogramLogCuda = jointHistogramLogCudaVecs[t].data().get();
        double *jointHistogramProCuda = jointHistogramProCudaVecs[t].data().get();
        // Define warped image texture
        auto warpedImageTexturePtr = Cuda::CreateTextureObject(warpedImageCuda + t * voxelNumber, cudaResourceTypeLinear,
                                                               voxelNumber * sizeof(float), cudaChannelFormatKindFloat, 1);
        auto warpedImageTexture = *warpedImageTexturePtr;
        // Fill the joint histograms
        if (approximation == false) {
            // No approximation is used for the Parzen windowing
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned>(0), activeVoxelNumber, [=]__device__(const unsigned index) {
                const int& voxel = tex1Dfetch<int>(maskTexture, index);
                const float& warValue = tex1Dfetch<float>(warpedImageTexture, voxel);
                if (warValue != warValue) return;
                auto&& [x, y, z] = reg_indexToDims_cuda(voxel, referenceImageDims);
                const float& refValue = tex3D<float>(referenceImageTexture, x, y, z);
                if (refValue != refValue) return;
                for (int r = int(refValue - 1); r < int(refValue + 3); r++) {
                    if (0 <= r && r < curRefBinNumber) {
                        const double& refBasis = GetBasisSplineValue<double>(refValue - r);
                        for (int w = int(warValue - 1); w < int(warValue + 3); w++) {
                            if (0 <= w && w < curFloBinNumber) {
                                const double& warBasis = GetBasisSplineValue<double>(warValue - w);
                                atomicAdd(&jointHistogramProCuda[r + w * curRefBinNumber], refBasis * warBasis);
                            }
                        }
                    }
                }
            });
        } else {
            // An approximation is used for the Parzen windowing. First intensities are binarised then
            // the histogram is convolved with a spine kernel function.
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned>(0), activeVoxelNumber, [=]__device__(const unsigned index) {
                const int& voxel = tex1Dfetch<int>(maskTexture, index);
                const float& warValue = tex1Dfetch<float>(warpedImageTexture, voxel);
                if (warValue != warValue) return;
                auto&& [x, y, z] = reg_indexToDims_cuda(voxel, referenceImageDims);
                const float& refValue = tex3D<float>(referenceImageTexture, x, y, z);
                if (refValue != refValue) return;
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
        const double& activeVoxel = thrust::reduce(thrust::device, jointHistogramProCudaVecs[t].begin(), jointHistogramProCudaVecs[t].end(), 0.0, thrust::plus<double>());
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
            const double& valPro = jointHistogramProCuda[curRefBinNumber * curFloBinNumber + r];
            if (valPro > 0) {
                const double& valLog = log(valPro);
                jointHistogramLogCuda[curRefBinNumber * curFloBinNumber + r] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
        // Compute the entropy of the warped floating image
        it = thrust::counting_iterator<unsigned short>(0);
        entropyValues[t][1] = thrust::transform_reduce(thrust::device, it, it + curFloBinNumber, [=]__device__(const unsigned short f) {
            const double& valPro = jointHistogramProCuda[curRefBinNumber * curFloBinNumber + curRefBinNumber + f];
            if (valPro > 0) {
                const double& valLog = log(valPro);
                jointHistogramLogCuda[curRefBinNumber * curFloBinNumber + curRefBinNumber + f] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
        // Compute the joint entropy
        it = thrust::counting_iterator<unsigned short>(0);
        entropyValues[t][2] = thrust::transform_reduce(thrust::device, it, it + curRefBinNumber * curFloBinNumber, [=]__device__(const unsigned short index) {
            const double& valPro = jointHistogramProCuda[index];
            if (valPro > 0) {
                const double& valLog = log(valPro);
                jointHistogramLogCuda[index] = valLog;
                return -valPro * valLog;
            } else return 0.0;
        }, 0.0, thrust::plus<double>());
    } // iterate over all time point in the reference image
}
/* *************************************************************** */
static double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                        const cudaArray *referenceImageCuda,
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
/// Called when we only have one target and one source image
void reg_getVoxelBasedNmiGradient_gpu(const nifti_image *referenceImage,
                                      const cudaArray *referenceImageCuda,
                                      const float *warpedImageCuda,
                                      const float4 *warpedGradientCuda,
                                      const float *logJointHistogramCuda,
                                      float4 *voxelBasedGradientCuda,
                                      const int *maskCuda,
                                      const size_t activeVoxelNumber,
                                      const double *entropies,
                                      const int refBinning,
                                      const int floBinning) {
    auto blockSize = CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const int3 imageSize = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int binNumber = refBinning * floBinning + refBinning + floBinning;
    const float normalisedJE = (float)(entropies[2] * entropies[3]);
    const float nmi = (float)((entropies[0] + entropies[1]) / entropies[2]);

    auto referenceImageTexture = Cuda::CreateTextureObject(referenceImageCuda, cudaResourceTypeArray, 0,
                                                           cudaChannelFormatKindNone, 1, cudaFilterModePoint, true);
    auto warpedImageTexture = Cuda::CreateTextureObject(warpedImageCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                        cudaChannelFormatKindFloat, 1);
    auto warpedGradientTexture = Cuda::CreateTextureObject(warpedGradientCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float4),
                                                           cudaChannelFormatKindFloat, 4);
    auto histogramTexture = Cuda::CreateTextureObject(logJointHistogramCuda, cudaResourceTypeLinear, binNumber * sizeof(float),
                                                      cudaChannelFormatKindFloat, 1);
    auto maskTexture = Cuda::CreateTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                 cudaChannelFormatKindSigned, 1);

    if (referenceImage->nz > 1) {
        const unsigned blocks = blockSize->reg_getVoxelBasedNmiGradientUsingPw3D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getVoxelBasedNmiGradientUsingPw3D_kernel<<<gridDims, blockDims>>>(voxelBasedGradientCuda, *referenceImageTexture, *warpedImageTexture,
                                                                              *warpedGradientTexture, *histogramTexture, *maskTexture,
                                                                              imageSize, refBinning, floBinning, normalisedJE, nmi,
                                                                              (unsigned)activeVoxelNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_getVoxelBasedNmiGradientUsingPw2D;
        const unsigned grids = (unsigned)Ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getVoxelBasedNmiGradientUsingPw2D_kernel<<<gridDims, blockDims>>>(voxelBasedGradientCuda, *referenceImageTexture, *warpedImageTexture,
                                                                              *warpedGradientTexture, *histogramTexture, *maskTexture,
                                                                              imageSize, refBinning, floBinning, normalisedJE, nmi,
                                                                              (unsigned)activeVoxelNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    // Call compute similarity measure to calculate joint histogram
    this->GetSimilarityMeasureValue();

    // The latest joint histogram is transferred onto the GPU
    thrust::device_vector<float> jointHistogramLogCuda(this->jointHistogramLog[0], this->jointHistogramLog[0] + this->totalBinNumber[0]);

    // The gradient of the NMI is computed on the GPU
    reg_getVoxelBasedNmiGradient_gpu(this->referenceImage,
                                     this->referenceImageCuda,
                                     this->warpedImageCuda,
                                     this->warpedGradientCuda,
                                     jointHistogramLogCuda.data().get(),
                                     this->voxelBasedGradientCuda,
                                     this->referenceMaskCuda,
                                     this->activeVoxelNumber,
                                     this->entropyValues[0],
                                     this->referenceBinNumber[0],
                                     this->floatingBinNumber[0]);
}
/* *************************************************************** */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    // The latest joint histogram is transferred onto the GPU
    thrust::device_vector<float> jointHistogramLogCudaBw(this->jointHistogramLogBw[0], this->jointHistogramLogBw[0] + this->totalBinNumber[0]);

    // The gradient of the NMI is computed on the GPU
    reg_getVoxelBasedNmiGradient_gpu(this->floatingImage,
                                     this->floatingImageCuda,
                                     this->warpedImageBwCuda,
                                     this->warpedGradientBwCuda,
                                     jointHistogramLogCudaBw.data().get(),
                                     this->voxelBasedGradientBwCuda,
                                     this->floatingMaskCuda,
                                     this->activeVoxelNumber,
                                     this->entropyValuesBw[0],
                                     this->floatingBinNumber[0],
                                     this->referenceBinNumber[0]);
}
/* *************************************************************** */
