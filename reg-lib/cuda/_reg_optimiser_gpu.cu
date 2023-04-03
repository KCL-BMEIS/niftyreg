#include "_reg_optimiser_gpu.h"
#include "_reg_optimiser_kernels.cu"

/* *************************************************************** */
reg_optimiser_gpu::reg_optimiser_gpu(): reg_optimiser<float>::reg_optimiser() {
    this->currentDofCuda = nullptr;
    this->bestDofCuda = nullptr;
    this->gradientCuda = nullptr;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_optimiser_gpu::reg_optimiser_gpu() called\n");
#endif
}
/* *************************************************************** */
reg_optimiser_gpu::~reg_optimiser_gpu() {
    if (this->bestDofCuda) {
        cudaCommon_free(this->bestDofCuda);
        this->bestDofCuda = nullptr;
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_optimiser_gpu::~reg_optimiser_gpu() called\n");
#endif
}
/* *************************************************************** */
void reg_optimiser_gpu::Initialise(size_t nvox,
                                   int ndim,
                                   bool optX,
                                   bool optY,
                                   bool optZ,
                                   size_t maxIt,
                                   size_t startIt,
                                   InterfaceOptimiser *intOpt,
                                   float *cppData,
                                   float *gradData,
                                   size_t nvoxBw,
                                   float *cppDataBw,
                                   float *gradDataBw) {
    this->dofNumber = nvox;
    this->ndim = ndim;
    this->optimiseX = optX;
    this->optimiseY = optY;
    this->optimiseZ = optZ;
    this->maxIterationNumber = maxIt;
    this->currentIterationNumber = startIt;

    // Arrays are converted from float to float4
    this->currentDofCuda = reinterpret_cast<float4*>(cppData);

    if (gradData)
        this->gradientCuda = reinterpret_cast<float4*>(gradData);

    if (this->bestDofCuda)
        cudaCommon_free(this->bestDofCuda);

    if (cudaCommon_allocateArrayToDevice(&this->bestDofCuda, (int)(this->GetVoxNumber()))) {
        printf("[NiftyReg ERROR] Error when allocating the best control point array on the GPU.\n");
        reg_exit();
    }

    this->StoreCurrentDof();

    this->intOpt = intOpt;
    this->bestObjFunctionValue = this->currentObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_optimiser_gpu::Initialise() called\n");
#endif
}
/* *************************************************************** */
void reg_optimiser_gpu::RestoreBestDof() {
    // restore forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentDofCuda,
                                 this->bestDofCuda,
                                 this->GetVoxNumber() * sizeof(float4),
                                 cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_optimiser_gpu::StoreCurrentDof() {
    // Store forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestDofCuda,
                                 this->currentDofCuda,
                                 this->GetVoxNumber() * sizeof(float4),
                                 cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_optimiser_gpu::Perturbation(float length) {
    // TODO: Implement reg_optimiser_gpu::Perturbation()
}
/* *************************************************************** */
reg_conjugateGradient_gpu::reg_conjugateGradient_gpu(): reg_optimiser_gpu::reg_optimiser_gpu() {
    this->array1 = nullptr;
    this->array2 = nullptr;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_conjugateGradient_gpu::reg_conjugateGradient_gpu() called\n");
#endif
}
/* *************************************************************** */
reg_conjugateGradient_gpu::~reg_conjugateGradient_gpu() {
    if (this->array1) {
        cudaCommon_free(this->array1);
        this->array1 = nullptr;
    }

    if (this->array2) {
        cudaCommon_free(this->array2);
        this->array2 = nullptr;
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_conjugateGradient_gpu::~reg_conjugateGradient_gpu() called\n");
#endif
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::Initialise(size_t nvox,
                                           int ndim,
                                           bool optX,
                                           bool optY,
                                           bool optZ,
                                           size_t maxIt,
                                           size_t startIt,
                                           InterfaceOptimiser *intOpt,
                                           float *cppData,
                                           float *gradData,
                                           size_t nvoxBw,
                                           float *cppDataBw,
                                           float *gradDataBw) {
    reg_optimiser_gpu::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData);
    this->firstCall = true;
    if (cudaCommon_allocateArrayToDevice<float4>(&this->array1, (int)(this->GetVoxNumber()))) {
        printf("[NiftyReg ERROR] Error when allocating the first conjugate gradient array on the GPU.\n");
        reg_exit();
    }
    if (cudaCommon_allocateArrayToDevice<float4>(&this->array2, (int)(this->GetVoxNumber()))) {
        printf("[NiftyReg ERROR] Error when allocating the second conjugate gradient array on the GPU.\n");
        reg_exit();
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_conjugateGradient_gpu::Initialise() called\n");
#endif
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::UpdateGradientValues() {
    if (this->firstCall) {
        reg_initialiseConjugateGradient_gpu(this->gradientCuda,
                                            this->array1,
                                            this->array2,
                                            this->GetVoxNumber());
        this->firstCall = false;
    } else {
        reg_GetConjugateGradient_gpu(this->gradientCuda,
                                     this->array1,
                                     this->array2,
                                     this->GetVoxNumber());
    }
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::Optimise(float maxLength,
                                         float smallLength,
                                         float &startLength) {
    this->UpdateGradientValues();
    reg_optimiser::Optimise(maxLength,
                            smallLength,
                            startLength);
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::Perturbation(float length) {
    reg_optimiser_gpu::Perturbation(length);
    this->firstCall = true;
}
/* *************************************************************** */
void reg_initialiseConjugateGradient_gpu(float4 *gradientImageCuda,
                                         float4 *conjugateGCuda,
                                         float4 *conjugateHCuda,
                                         const size_t& nVoxels) {
    auto gradientImageTexture = cudaCommon_createTextureObject(gradientImageCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                               cudaChannelFormatKindFloat, 4, cudaFilterModePoint);

    const unsigned blocks = (unsigned)NiftyReg::CudaContext::GetBlockSize()->reg_initialiseConjugateGradient;
    const unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);

    reg_initialiseConjugateGradient_kernel<<<gridDims, blockDims>>>(conjugateGCuda, *gradientImageTexture, (unsigned)nVoxels);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    NR_CUDA_SAFE_CALL(cudaMemcpy(conjugateHCuda, conjugateGCuda, nVoxels * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_GetConjugateGradient_gpu(float4 *gradientImageCuda,
                                  float4 *conjugateGCuda,
                                  float4 *conjugateHCuda,
                                  const size_t& nVoxels) {
    auto gradientImageTexture = cudaCommon_createTextureObject(gradientImageCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                               cudaChannelFormatKindFloat, 4, cudaFilterModePoint);
    auto conjugateGTexture = cudaCommon_createTextureObject(conjugateGCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                            cudaChannelFormatKindFloat, 4, cudaFilterModePoint);
    auto conjugateHTexture = cudaCommon_createTextureObject(conjugateHCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                            cudaChannelFormatKindFloat, 4, cudaFilterModePoint);

    // gam = sum((grad+g)*grad)/sum(HxG);
    unsigned blocks = NiftyReg::CudaContext::GetBlockSize()->reg_GetConjugateGradient1;
    unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    dim3 blockDims(blocks, 1, 1);
    dim3 gridDims(grids, grids, 1);

    float2 *sumsCuda;
    NR_CUDA_SAFE_CALL(cudaMalloc(&sumsCuda, nVoxels * sizeof(float2)));
    reg_GetConjugateGradient1_kernel<<<gridDims, blockDims>>>(sumsCuda, *gradientImageTexture, *conjugateGTexture, *conjugateHTexture, (unsigned)nVoxels);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    float2 *sums;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&sums, nVoxels * sizeof(float2)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(sums, sumsCuda, nVoxels * sizeof(float2), cudaMemcpyDeviceToHost));
    NR_CUDA_SAFE_CALL(cudaFree(sumsCuda));
    double dgg = 0;
    double gg = 0;
    for (size_t i = 0; i < nVoxels; i++) {
        dgg += sums[i].x;
        gg += sums[i].y;
    }
    const float gam = (float)(dgg / gg);
    NR_CUDA_SAFE_CALL(cudaFreeHost(sums));

    blocks = (unsigned)NiftyReg::CudaContext::GetBlockSize()->reg_GetConjugateGradient2;
    grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    gridDims = dim3(blocks, 1, 1);
    blockDims = dim3(grids, grids, 1);
    reg_GetConjugateGradient2_kernel<<<blockDims, gridDims>>>(gradientImageCuda, conjugateGCuda, conjugateHCuda, (unsigned)nVoxels, gam);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
void reg_updateControlPointPosition_gpu(const size_t& nVoxels,
                                        float4 *controlPointImageCuda,
                                        const float4 *bestControlPointCuda,
                                        const float4 *gradientImageCuda,
                                        const float& scale,
                                        const bool& optimiseX,
                                        const bool& optimiseY,
                                        const bool& optimiseZ) {
    auto bestControlPointTexture = cudaCommon_createTextureObject(bestControlPointCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                                  cudaChannelFormatKindFloat, 4, cudaFilterModePoint);
    auto gradientImageTexture = cudaCommon_createTextureObject(gradientImageCuda, cudaResourceTypeLinear, false, nVoxels * sizeof(float4),
                                                               cudaChannelFormatKindFloat, 4, cudaFilterModePoint);

    const unsigned blocks = (unsigned)NiftyReg::CudaContext::GetBlockSize()->reg_updateControlPointPosition;
    const unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    const dim3 blockDims(blocks, 1, 1);
    const dim3 gridDims(grids, grids, 1);
    reg_updateControlPointPosition_kernel<<<gridDims, blockDims>>>(controlPointImageCuda, *bestControlPointTexture, *gradientImageTexture, (unsigned)nVoxels, scale, optimiseX, optimiseY, optimiseZ);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
