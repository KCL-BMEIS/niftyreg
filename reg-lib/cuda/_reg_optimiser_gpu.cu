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
void reg_conjugateGradient_gpu::reg_test_optimiser() {
    this->UpdateGradientValues();
    reg_optimiser_gpu::reg_test_optimiser();
}
/* *************************************************************** */
void reg_initialiseConjugateGradient_gpu(float4 *gradientArray_d,
                                         float4 *conjugateG_d,
                                         float4 *conjugateH_d,
                                         int nodeNumber) {
    // Get the BlockSize - The values have been set in CudaContextSingleton
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::GetInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber, &nodeNumber, sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, gradientArray_d, nodeNumber * sizeof(float4)));

    const unsigned int Grid_reg_initialiseConjugateGradient =
        (unsigned int)reg_ceil(sqrtf((float)nodeNumber / (float)NR_BLOCK->Block_reg_initialiseConjugateGradient));
    dim3 G1(Grid_reg_initialiseConjugateGradient, Grid_reg_initialiseConjugateGradient, 1);
    dim3 B1(NR_BLOCK->Block_reg_initialiseConjugateGradient, 1, 1);

    reg_initialiseConjugateGradient_kernel <<< G1, B1 >>> (conjugateG_d);
    NR_CUDA_CHECK_KERNEL(G1, B1);
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture));
    NR_CUDA_SAFE_CALL(cudaMemcpy(conjugateH_d, conjugateG_d, nodeNumber * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_GetConjugateGradient_gpu(float4 *gradientArray_d,
                                  float4 *conjugateG_d,
                                  float4 *conjugateH_d,
                                  int nodeNumber) {
    // Get the BlockSize - The values have been set in CudaContextSingleton
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::GetInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber, &nodeNumber, sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, conjugateGTexture, conjugateG_d, nodeNumber * sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, conjugateHTexture, conjugateH_d, nodeNumber * sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, gradientArray_d, nodeNumber * sizeof(float4)));

    // gam = sum((grad+g)*grad)/sum(HxG);
    const unsigned int Grid_reg_GetConjugateGradient1 = (unsigned int)reg_ceil(sqrtf((float)nodeNumber / (float)NR_BLOCK->Block_reg_GetConjugateGradient1));
    dim3 B1(NR_BLOCK->Block_reg_GetConjugateGradient1, 1, 1);
    dim3 G1(Grid_reg_GetConjugateGradient1, Grid_reg_GetConjugateGradient1, 1);

    float2 *sum_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&sum_d, nodeNumber * sizeof(float2)));
    reg_GetConjugateGradient1_kernel <<< G1, B1 >>> (sum_d);
    NR_CUDA_CHECK_KERNEL(G1, B1);
    float2 *sum_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&sum_h, nodeNumber * sizeof(float2)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(sum_h, sum_d, nodeNumber * sizeof(float2), cudaMemcpyDeviceToHost));
    NR_CUDA_SAFE_CALL(cudaFree(sum_d));
    double dgg = 0;
    double gg = 0;
    for (int i = 0; i < nodeNumber; i++) {
        dgg += sum_h[i].x;
        gg += sum_h[i].y;
    }
    float gam = (float)(dgg / gg);
    NR_CUDA_SAFE_CALL(cudaFreeHost(sum_h));

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor, &gam, sizeof(float)));
    const unsigned int Grid_reg_GetConjugateGradient2 = (unsigned int)reg_ceil(sqrtf((float)nodeNumber / (float)NR_BLOCK->Block_reg_GetConjugateGradient2));
    dim3 B2(NR_BLOCK->Block_reg_GetConjugateGradient2, 1, 1);
    dim3 G2(Grid_reg_GetConjugateGradient2, Grid_reg_GetConjugateGradient2, 1);
    reg_GetConjugateGradient2_kernel <<< G2, B2 >>> (gradientArray_d, conjugateG_d, conjugateH_d);
    NR_CUDA_CHECK_KERNEL(G1, B1);

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(conjugateGTexture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(conjugateHTexture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture));

}
/* *************************************************************** */
void reg_updateControlPointPosition_gpu(const nifti_image *controlPointImage,
                                        float4 *controlPointImageArray_d,
                                        const float4 *bestControlPointPosition_d,
                                        const float4 *gradientArray_d,
                                        const float& currentLength) {
    // Get the BlockSize - The values have been set in CudaContextSingleton
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::GetInstance(0);

    const int nodeNumber = CalcVoxelNumber(*controlPointImage);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber, &nodeNumber, sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor, &currentLength, sizeof(float)));

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, bestControlPointPosition_d, nodeNumber * sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, gradientArray_d, nodeNumber * sizeof(float4)));

    const unsigned int Grid_reg_updateControlPointPosition =
        (unsigned int)reg_ceil(sqrtf((float)nodeNumber / (float)NR_BLOCK->Block_reg_updateControlPointPosition));
    dim3 B1(NR_BLOCK->Block_reg_updateControlPointPosition, 1, 1);
    dim3 G1(Grid_reg_updateControlPointPosition, Grid_reg_updateControlPointPosition, 1);
    reg_updateControlPointPosition_kernel <<< G1, B1 >>> (controlPointImageArray_d);
    NR_CUDA_CHECK_KERNEL(G1, B1);
    // Unbind the textures
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture));
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_updateControlPointPosition_gpu() called\n");
#endif
}
/* *************************************************************** */
