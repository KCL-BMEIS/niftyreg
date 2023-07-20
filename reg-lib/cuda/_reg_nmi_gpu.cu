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
#include <thrust/device_vector.h>

/* *************************************************************** */
reg_nmi_gpu::reg_nmi_gpu(): reg_nmi::reg_nmi() {
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi_gpu constructor called");
#endif
}
/* *************************************************************** */
reg_nmi_gpu::~reg_nmi_gpu() {
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi_gpu destructor called");
#endif
}
/* *************************************************************** */
void reg_nmi_gpu::InitialiseMeasure(nifti_image *refImg, cudaArray *refImgCuda,
                                    nifti_image *floImg, cudaArray *floImgCuda,
                                    int *refMask, int *refMaskCuda,
                                    size_t activeVoxNum,
                                    nifti_image *warpedImg, float *warpedImgCuda,
                                    nifti_image *warpedGrad, float4 *warpedGradCuda,
                                    nifti_image *voxelBasedGrad, float4 *voxelBasedGradCuda,
                                    nifti_image *localWeightSim,
                                    int *floMask, int *floMaskCuda,
                                    nifti_image *warpedImgBw, float *warpedImgBwCuda,
                                    nifti_image *warpedGradBw, float4 *warpedGradBwCuda,
                                    nifti_image *voxelBasedGradBw, float4 *voxelBasedGradBwCuda) {
    this->DeallocateHistogram();
    reg_nmi::InitialiseMeasure(refImg, floImg, refMask, warpedImg, warpedGrad, voxelBasedGrad,
                               localWeightSim, floMask, warpedImgBw, warpedGradBw, voxelBasedGradBw);
    reg_measure_gpu::InitialiseMeasure(refImg, refImgCuda, floImg, floImgCuda, refMask, refMaskCuda, activeVoxNum, warpedImg, warpedImgCuda,
                                       warpedGrad, warpedGradCuda, voxelBasedGrad, voxelBasedGradCuda, localWeightSim, floMask, floMaskCuda,
                                       warpedImgBw, warpedImgBwCuda, warpedGradBw, warpedGradBwCuda, voxelBasedGradBw, voxelBasedGradBwCuda);
    // Check if the input images have multiple timepoints
    if (this->referenceTimePoint > 1 || this->floatingImage->nt > 1) {
        reg_print_fct_error("reg_nmi_gpu::InitialiseMeasure");
        reg_print_msg_error("Multiple timepoints are not yet supported");
        reg_exit();
    }
    // The reference and floating images have to be updated on the device
    if (cudaCommon_transferNiftiToArrayOnDevice<float>(this->referenceImageCuda, this->referenceImage) ||
        cudaCommon_transferNiftiToArrayOnDevice<float>(this->floatingImageCuda, this->floatingImage)) {
        reg_print_fct_error("reg_nmi_gpu::InitialiseMeasure");
        reg_print_msg_error("Error when transferring the reference or floating image");
        reg_exit();
    }
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi_gpu::InitialiseMeasure called");
#endif
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValue() {
    // The NMI computation is performed into the host for now
    // The relevant images have to be transferred from the device to the host
    cudaCommon_transferFromDeviceToNifti<float>(this->warpedImage, this->warpedImageCuda);
    reg_getNMIValue<float>(this->referenceImage,
                           this->warpedImage,
                           this->timePointWeight,
                           this->referenceBinNumber,
                           this->floatingBinNumber,
                           this->totalBinNumber,
                           this->jointHistogramLog,
                           this->jointHistogramPro,
                           this->entropyValues,
                           this->referenceMask);

    if (this->isSymmetric) {
        cudaCommon_transferFromDeviceToNifti<float>(this->warpedImageBw, this->warpedImageBwCuda);
        reg_getNMIValue<float>(this->floatingImage,
                               this->warpedImageBw,
                               this->timePointWeight,
                               this->floatingBinNumber,
                               this->referenceBinNumber,
                               this->totalBinNumber,
                               this->jointHistogramLogBw,
                               this->jointHistogramProBw,
                               this->entropyValuesBw,
                               this->floatingMask);
    }

    double nmiFw = 0, nmiBw = 0;
    for (int t = 0; t < this->referenceTimePoint; ++t) {
        if (this->timePointWeight[t] > 0) {
            nmiFw += timePointWeight[t] * (this->entropyValues[t][0] + this->entropyValues[t][1]) / this->entropyValues[t][2];
            if (this->isSymmetric)
                nmiBw += timePointWeight[t] * (this->entropyValuesBw[t][0] + this->entropyValuesBw[t][1]) / this->entropyValuesBw[t][2];
        }
    }

#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi_gpu::GetSimilarityMeasureValue called");
#endif
    return nmiFw + nmiBw;
}
/* *************************************************************** */
/// Called when we only have one target and one source image
void reg_getVoxelBasedNMIGradient_gpu(const nifti_image *referenceImage,
                                      const cudaArray *referenceImageCuda,
                                      const float *warpedImageCuda,
                                      const float4 *warpedGradientCuda,
                                      const float *logJointHistogramCuda,
                                      float4 *voxelBasedGradientCuda,
                                      const int *maskCuda,
                                      const size_t& activeVoxelNumber,
                                      const double *entropies,
                                      const int& refBinning,
                                      const int& floBinning) {
    auto blockSize = NiftyReg::CudaContext::GetBlockSize();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    const int3 imageSize = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int binNumber = refBinning * floBinning + refBinning + floBinning;
    const float normalisedJE = (float)(entropies[2] * entropies[3]);
    const float nmi = (float)((entropies[0] + entropies[1]) / entropies[2]);

    auto referenceImageTexture = cudaCommon_createTextureObject(referenceImageCuda, cudaResourceTypeArray, 0,
                                                                cudaChannelFormatKindNone, 1, cudaFilterModePoint, true);
    auto warpedImageTexture = cudaCommon_createTextureObject(warpedImageCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float),
                                                             cudaChannelFormatKindFloat, 1);
    auto warpedGradientTexture = cudaCommon_createTextureObject(warpedGradientCuda, cudaResourceTypeLinear, voxelNumber * sizeof(float4),
                                                                cudaChannelFormatKindFloat, 4);
    auto histogramTexture = cudaCommon_createTextureObject(logJointHistogramCuda, cudaResourceTypeLinear, binNumber * sizeof(float),
                                                           cudaChannelFormatKindFloat, 1);
    auto maskTexture = cudaCommon_createTextureObject(maskCuda, cudaResourceTypeLinear, activeVoxelNumber * sizeof(int),
                                                      cudaChannelFormatKindSigned, 1);
    NR_CUDA_SAFE_CALL(cudaMemset(voxelBasedGradientCuda, 0, voxelNumber * sizeof(float4)));

    if (referenceImage->nz > 1) {
        const unsigned blocks = blockSize->reg_getVoxelBasedNMIGradientUsingPW3D;
        const unsigned grids = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getVoxelBasedNMIGradientUsingPW3D_kernel<<<gridDims, blockDims>>>(voxelBasedGradientCuda, *referenceImageTexture, *warpedImageTexture,
                                                                              *warpedGradientTexture, *histogramTexture, *maskTexture,
                                                                              imageSize, refBinning, floBinning, normalisedJE, nmi,
                                                                              (unsigned)activeVoxelNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    } else {
        const unsigned blocks = blockSize->reg_getVoxelBasedNMIGradientUsingPW2D;
        const unsigned grids = (unsigned)ceil(sqrtf((float)activeVoxelNumber / (float)blocks));
        const dim3 gridDims(grids, grids, 1);
        const dim3 blockDims(blocks, 1, 1);
        reg_getVoxelBasedNMIGradientUsingPW2D_kernel<<<gridDims, blockDims>>>(voxelBasedGradientCuda, *referenceImageTexture, *warpedImageTexture,
                                                                              *warpedGradientTexture, *histogramTexture, *maskTexture,
                                                                              imageSize, refBinning, floBinning, normalisedJE, nmi,
                                                                              (unsigned)activeVoxelNumber);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(currentTimepoint);
    if (this->timePointWeight[currentTimepoint] == 0)
        return;

    // Call compute similarity measure to calculate joint histogram
    this->GetSimilarityMeasureValue();

    // The latest joint histogram is transferred onto the GPU
    thrust::device_vector<float> jointHistogramLogCuda(this->jointHistogramLog[0], this->jointHistogramLog[0] + this->totalBinNumber[0]);

    // The gradient of the NMI is computed on the GPU
    reg_getVoxelBasedNMIGradient_gpu(this->referenceImage,
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

    if (this->isSymmetric) {
        thrust::device_vector<float> jointHistogramLogCudaBw(this->jointHistogramLogBw[0], this->jointHistogramLogBw[0] + this->totalBinNumber[0]);
        reg_getVoxelBasedNMIGradient_gpu(this->floatingImage,
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
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradient called\n");
#endif
}
/* *************************************************************** */
