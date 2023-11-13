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
    this->DeallocateHistogram();
    reg_nmi::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum,
                                       warpedImg, warpedImgCuda, warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda,
                                       localWeightSim, localWeightSimCuda, floMask, floMaskCuda, warpedImgBw, warpedImgBwCuda,
                                       warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda);
    // Check if the input images have multiple timepoints
    if (this->referenceTimePoints > 1 || this->floatingImage->nt > 1)
        NR_FATAL_ERROR("Multiple timepoints are not yet supported");
    // The reference and floating images have to be updated on the device
    Cuda::TransferNiftiToDevice<float>(this->referenceImageCuda, this->referenceImage);
    Cuda::TransferNiftiToDevice<float>(this->floatingImageCuda, this->floatingImage);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 nifti_image *warpedImage,
                                 const float *warpedImageCuda,
                                 const double *timePointWeights,
                                 const unsigned short *referenceBinNumber,
                                 const unsigned short *floatingBinNumber,
                                 const unsigned short *totalBinNumber,
                                 double **jointHistogramLog,
                                 double **jointHistogramPro,
                                 double **entropyValues,
                                 const int *referenceMask,
                                 const int referenceTimePoints,
                                 const bool approximation) {
    // TODO: Implement the NMI computation for CUDA
    // The NMI computation is performed on the host for now
    Cuda::TransferFromDeviceToNifti<float>(warpedImage, warpedImageCuda);
    reg_getNmiValue<float>(referenceImage,
                           warpedImage,
                           timePointWeights,
                           referenceTimePoints,
                           referenceBinNumber,
                           floatingBinNumber,
                           totalBinNumber,
                           jointHistogramLog,
                           jointHistogramPro,
                           entropyValues,
                           referenceMask,
                           approximation);

    double nmi = 0;
    for (int t = 0; t < referenceTimePoints; ++t) {
        if (timePointWeights[t] > 0)
            nmi += timePointWeights[t] * (entropyValues[t][0] + entropyValues[t][1]) / entropyValues[t][2];
    }
    return nmi;
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->warpedImageCuda,
                                       this->timePointWeights,
                                       this->referenceBinNumber,
                                       this->floatingBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLog,
                                       this->jointHistogramPro,
                                       this->entropyValues,
                                       this->referenceMask,
                                       this->referenceTimePoints,
                                       this->approximatePw);
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->warpedImageBwCuda,
                                       this->timePointWeights,
                                       this->floatingBinNumber,
                                       this->referenceBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLogBw,
                                       this->jointHistogramProBw,
                                       this->entropyValuesBw,
                                       this->floatingMask,
                                       this->referenceTimePoints,
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
