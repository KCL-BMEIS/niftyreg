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

#include "_reg_nmi.h"
#include "_reg_nmi_gpu.h"
#include "_reg_nmi_kernels.cu"

/* *************************************************************** */
reg_nmi_gpu::reg_nmi_gpu(): reg_nmi::reg_nmi() {
    this->forwardJointHistogramLog_device = nullptr;
    //	this->backwardJointHistogramLog_device=nullptr;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu constructor called\n");
#endif
}
/* *************************************************************** */
reg_nmi_gpu::~reg_nmi_gpu() {
    this->DeallocateHistogram();
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu destructor called\n");
#endif
}
/* *************************************************************** */
void reg_nmi_gpu::DeallocateHistogram() {
    if (this->forwardJointHistogramLog_device != nullptr) {
        cudaFree(this->forwardJointHistogramLog_device);
        this->forwardJointHistogramLog_device = nullptr;
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu::DeallocateHistogram() called\n");
#endif
}
/* *************************************************************** */
void reg_nmi_gpu::InitialiseMeasure(nifti_image *refImg,
                                    nifti_image *floImg,
                                    int *refMask,
                                    size_t activeVoxNum,
                                    nifti_image *warpedImg,
                                    nifti_image *warpedGrad,
                                    nifti_image *voxelBasedGrad,
                                    nifti_image *localWeightSim,
                                    cudaArray *refImgCuda,
                                    cudaArray *floImgCuda,
                                    int *refMaskCuda,
                                    float *warpedImgCuda,
                                    float4 *warpedGradCuda,
                                    float4 *voxelBasedGradCuda) {
    this->DeallocateHistogram();
    reg_nmi::InitialiseMeasure(refImg,
                               floImg,
                               refMask,
                               warpedImg,
                               warpedGrad,
                               voxelBasedGrad);
    // Check if a symmetric measure is required
    if (this->isSymmetric) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] Symmetric scheme is not yet supported on the GPU\n");
        reg_exit();
    }
    // Check if the input images have multiple timepoints
    if (this->referenceTimePoint > 1 || this->floatingImage->nt > 1) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] Multiple timepoints are not yet supported on the GPU\n");
        reg_exit();
    }
    // Check that the input image are of type float
    if (this->referenceImage->datatype != NIFTI_TYPE_FLOAT32 ||
        this->warpedImage->datatype != NIFTI_TYPE_FLOAT32) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr, "[NiftyReg ERROR] Only single precision is supported on the GPU\n");
        reg_exit();
    }
    // Bind the required pointers
    this->referenceImageCuda = refImgCuda;
    this->floatingImageCuda = floImgCuda;
    this->referenceMaskCuda = refMaskCuda;
    this->activeVoxelNumber = activeVoxNum;
    this->warpedImageCuda = warpedImgCuda;
    this->warpedGradientCuda = warpedGradCuda;
    this->voxelBasedGradientCuda = voxelBasedGradCuda;
    // The reference and floating images have to be updated on the device
    if (cudaCommon_transferNiftiToArrayOnDevice<float>(this->referenceImageCuda, this->referenceImage)) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        printf("[NiftyReg ERROR] Error when transferring the reference image.\n");
        reg_exit();
    }
    if (cudaCommon_transferNiftiToArrayOnDevice<float>(this->floatingImageCuda, this->floatingImage)) {
        fprintf(stderr, "[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        printf("[NiftyReg ERROR] Error when transferring the floating image.\n");
        reg_exit();
    }
    // Allocate the required joint histogram on the GPU
    cudaMalloc(&this->forwardJointHistogramLog_device, this->totalBinNumber[0] * sizeof(float));

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu::InitialiseMeasure called\n");
#endif
}
/* *************************************************************** */
double reg_nmi_gpu::GetSimilarityMeasureValue() {
    // The NMI computation is performed into the host for now
    // The relevant images have to be transferred from the device to the host
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->warpedImage->data,
                                 this->warpedImageCuda,
                                 this->warpedImage->nvox *
                                 this->warpedImage->nbyper,
                                 cudaMemcpyDeviceToHost));

    reg_getNMIValue<float>(this->referenceImage,
                           this->warpedImage,
                           this->timePointWeight,
                           this->referenceBinNumber,
                           this->floatingBinNumber,
                           this->totalBinNumber,
                           this->forwardJointHistogramLog,
                           this->forwardJointHistogramPro,
                           this->forwardEntropyValues,
                           this->referenceMask);

    const double nmi_value = (this->forwardEntropyValues[0][0] + this->forwardEntropyValues[0][1]) / this->forwardEntropyValues[0][2];

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu::GetSimilarityMeasureValue called\n");
#endif
    return nmi_value;
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
    // The latest joint histogram is transferred onto the GPU
    float *temp = (float*)malloc(this->totalBinNumber[0] * sizeof(float));
    for (unsigned short i = 0; i < this->totalBinNumber[0]; ++i)
        temp[i] = static_cast<float>(this->forwardJointHistogramLog[0][i]);
    cudaMemcpy(this->forwardJointHistogramLog_device,
               temp,
               this->totalBinNumber[0] * sizeof(float),
               cudaMemcpyHostToDevice);
    free(temp);

    // The gradient of the NMI is computed on the GPU
    reg_getVoxelBasedNMIGradient_gpu(this->referenceImage,
                                     this->referenceImageCuda,
                                     this->warpedImageCuda,
                                     this->warpedGradientCuda,
                                     this->forwardJointHistogramLog_device,
                                     this->voxelBasedGradientCuda,
                                     this->referenceMaskCuda,
                                     this->activeVoxelNumber,
                                     this->forwardEntropyValues[0],
                                     this->referenceBinNumber[0],
                                     this->floatingBinNumber[0]);
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradient called\n");
#endif
}
/* *************************************************************** */
