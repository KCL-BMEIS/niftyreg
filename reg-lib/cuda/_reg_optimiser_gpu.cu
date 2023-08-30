#include "_reg_optimiser_gpu.h"
#include "_reg_optimiser_kernels.cu"
#include "_reg_common_cuda_kernels.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

/* *************************************************************** */
reg_optimiser_gpu::reg_optimiser_gpu(): reg_optimiser<float>::reg_optimiser() {
    this->currentDofCuda = nullptr;
    this->currentDofBwCuda = nullptr;
    this->bestDofCuda = nullptr;
    this->bestDofBwCuda = nullptr;
    this->gradientCuda = nullptr;
    this->gradientBwCuda = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_optimiser_gpu::~reg_optimiser_gpu() {
    if (this->bestDofCuda) {
        Cuda::Free(this->bestDofCuda);
        this->bestDofCuda = nullptr;
    }
    if (this->bestDofBwCuda) {
        Cuda::Free(this->bestDofBwCuda);
        this->bestDofBwCuda = nullptr;
    }
    NR_FUNC_CALLED();
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
    this->currentDofCuda = reinterpret_cast<float4*>(cppData);
    this->gradientCuda = reinterpret_cast<float4*>(gradData);

    Cuda::Free(this->bestDofCuda);
    Cuda::Allocate(&this->bestDofCuda, this->GetVoxNumber());

    this->isSymmetric = nvoxBw > 0 && cppDataBw && gradDataBw;
    if (this->isSymmetric) {
        this->dofNumberBw = nvoxBw;
        this->currentDofBwCuda = reinterpret_cast<float4*>(cppDataBw);
        this->gradientBwCuda = reinterpret_cast<float4*>(gradDataBw);
        Cuda::Free(this->bestDofBwCuda);
        Cuda::Allocate(&this->bestDofBwCuda, this->GetVoxNumberBw());
    }

    this->StoreCurrentDof();

    this->intOpt = intOpt;
    this->bestObjFunctionValue = this->currentObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();

    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_optimiser_gpu::RestoreBestDof() {
    // Restore forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentDofCuda, this->bestDofCuda, this->GetVoxNumber() * sizeof(float4), cudaMemcpyDeviceToDevice));
    // Restore backward transformation if required
    if (this->isSymmetric)
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentDofBwCuda, this->bestDofBwCuda, this->GetVoxNumberBw() * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_optimiser_gpu::StoreCurrentDof() {
    // Store forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestDofCuda, this->currentDofCuda, this->GetVoxNumber() * sizeof(float4), cudaMemcpyDeviceToDevice));
    // Store backward transformation if required
    if (this->isSymmetric)
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestDofBwCuda, this->currentDofBwCuda, this->GetVoxNumberBw() * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void reg_optimiser_gpu::Perturbation(float length) {
    // TODO: Implement reg_optimiser_gpu::Perturbation()
}
/* *************************************************************** */
reg_conjugateGradient_gpu::reg_conjugateGradient_gpu(): reg_optimiser_gpu::reg_optimiser_gpu() {
    this->array1 = nullptr;
    this->array1Bw = nullptr;
    this->array2 = nullptr;
    this->array2Bw = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_conjugateGradient_gpu::~reg_conjugateGradient_gpu() {
    if (this->array1) {
        Cuda::Free(this->array1);
        this->array1 = nullptr;
    }
    if (this->array1Bw) {
        Cuda::Free(this->array1Bw);
        this->array1Bw = nullptr;
    }
    if (this->array2) {
        Cuda::Free(this->array2);
        this->array2 = nullptr;
    }
    if (this->array2Bw) {
        Cuda::Free(this->array2Bw);
        this->array2Bw = nullptr;
    }
    NR_FUNC_CALLED();
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
    reg_optimiser_gpu::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
    this->firstCall = true;
    Cuda::Free(this->array1); Cuda::Free(this->array2);
    Cuda::Allocate<float4>(&this->array1, this->GetVoxNumber());
    Cuda::Allocate<float4>(&this->array2, this->GetVoxNumber());
    if (this->isSymmetric) {
        Cuda::Free(this->array1Bw); Cuda::Free(this->array2Bw);
        Cuda::Allocate<float4>(&this->array1Bw, this->GetVoxNumberBw());
        Cuda::Allocate<float4>(&this->array2Bw, this->GetVoxNumberBw());
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::UpdateGradientValues() {
    if (this->firstCall) {
        reg_initialiseConjugateGradient_gpu(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber());
        if (this->isSymmetric)
            reg_initialiseConjugateGradient_gpu(this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
        this->firstCall = false;
    } else {
        reg_getConjugateGradient_gpu(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber(),
                                     this->isSymmetric, this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
    }
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::Optimise(float maxLength,
                                         float smallLength,
                                         float& startLength) {
    this->UpdateGradientValues();
    reg_optimiser::Optimise(maxLength, smallLength, startLength);
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
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, cudaResourceTypeLinear,
                                                          nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);

    const unsigned blocks = CudaContext::GetBlockSize()->reg_initialiseConjugateGradient;
    const unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    const dim3 gridDims(grids, grids, 1);
    const dim3 blockDims(blocks, 1, 1);

    reg_initialiseConjugateGradient_kernel<<<gridDims, blockDims>>>(conjugateGCuda, *gradientImageTexture, (unsigned)nVoxels);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    NR_CUDA_SAFE_CALL(cudaMemcpy(conjugateHCuda, conjugateGCuda, nVoxels * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
struct Float2Sum {
    __host__ __device__ double2 operator()(const float2& a, const float2& b) const {
        return make_double2((double)a.x + (double)b.x, (double)a.y + (double)b.y);
    }
};
/* *************************************************************** */
void reg_getConjugateGradient_gpu(float4 *gradientImageCuda,
                                  float4 *conjugateGCuda,
                                  float4 *conjugateHCuda,
                                  const size_t& nVoxels,
                                  const bool& isSymmetric,
                                  float4 *gradientImageBwCuda,
                                  float4 *conjugateGBwCuda,
                                  float4 *conjugateHBwCuda,
                                  const size_t& nVoxelsBw) {
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, cudaResourceTypeLinear,
                                                          nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);
    auto conjugateGTexture = Cuda::CreateTextureObject(conjugateGCuda, cudaResourceTypeLinear,
                                                       nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);
    auto conjugateHTexture = Cuda::CreateTextureObject(conjugateHCuda, cudaResourceTypeLinear,
                                                       nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);
    Cuda::UniqueTextureObjectPtr gradientImageBwTexture(nullptr, nullptr), conjugateGBwTexture(nullptr, nullptr), conjugateHBwTexture(nullptr, nullptr);
    if (isSymmetric) {
        gradientImageBwTexture = std::move(Cuda::CreateTextureObject(gradientImageBwCuda, cudaResourceTypeLinear,
                                                                     nVoxelsBw * sizeof(float4), cudaChannelFormatKindFloat, 4));
        conjugateGBwTexture = std::move(Cuda::CreateTextureObject(conjugateGBwCuda, cudaResourceTypeLinear,
                                                                  nVoxelsBw * sizeof(float4), cudaChannelFormatKindFloat, 4));
        conjugateHBwTexture = std::move(Cuda::CreateTextureObject(conjugateHBwCuda, cudaResourceTypeLinear,
                                                                  nVoxelsBw * sizeof(float4), cudaChannelFormatKindFloat, 4));
    }

    // gam = sum((grad+g)*grad)/sum(HxG);
    unsigned blocks = CudaContext::GetBlockSize()->reg_getConjugateGradient1;
    unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    dim3 blockDims(blocks, 1, 1);
    dim3 gridDims(grids, grids, 1);

    thrust::device_vector<float2> sumsCuda(nVoxels + nVoxels % 2);  // Make it even for thrust::inner_product
    reg_getConjugateGradient1_kernel<<<gridDims, blockDims>>>(sumsCuda.data().get(), *gradientImageTexture,
                                                              *conjugateGTexture, *conjugateHTexture, (unsigned)nVoxels);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    const size_t sumsSizeHalf = sumsCuda.size() / 2;
    const double2 gg = thrust::inner_product(sumsCuda.begin(), sumsCuda.begin() + sumsSizeHalf, sumsCuda.begin() + sumsSizeHalf,
                                             make_double2(0, 0), thrust::plus<double2>(), Float2Sum());
    float gam = static_cast<float>(gg.x / gg.y);
    if (isSymmetric) {
        grids = (unsigned)reg_ceil(sqrtf((float)nVoxelsBw / (float)blocks));
        gridDims = dim3(blocks, 1, 1);
        blockDims = dim3(grids, grids, 1);
        thrust::device_vector<float2> sumsBwCuda(nVoxelsBw + nVoxelsBw % 2);  // Make it even for thrust::inner_product
        reg_getConjugateGradient1_kernel<<<gridDims, blockDims>>>(sumsBwCuda.data().get(), *gradientImageBwTexture,
                                                                  *conjugateGBwTexture, *conjugateHBwTexture, (unsigned)nVoxelsBw);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
        const size_t sumsBwSizeHalf = sumsBwCuda.size() / 2;
        const double2 ggBw = thrust::inner_product(sumsBwCuda.begin(), sumsBwCuda.begin() + sumsBwSizeHalf, sumsBwCuda.begin() + sumsBwSizeHalf,
                                                   make_double2(0, 0), thrust::plus<double2>(), Float2Sum());
        gam = static_cast<float>((gg.x + ggBw.x) / (gg.y + ggBw.y));
    }

    blocks = (unsigned)CudaContext::GetBlockSize()->reg_getConjugateGradient2;
    grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    gridDims = dim3(blocks, 1, 1);
    blockDims = dim3(grids, grids, 1);
    reg_getConjugateGradient2_kernel<<<blockDims, gridDims>>>(gradientImageCuda, conjugateGCuda, conjugateHCuda, (unsigned)nVoxels, gam);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    if (isSymmetric) {
        grids = (unsigned)reg_ceil(sqrtf((float)nVoxelsBw / (float)blocks));
        gridDims = dim3(blocks, 1, 1);
        blockDims = dim3(grids, grids, 1);
        reg_getConjugateGradient2_kernel<<<blockDims, gridDims>>>(gradientImageBwCuda, conjugateGBwCuda, conjugateHBwCuda, (unsigned)nVoxelsBw, gam);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
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
    auto bestControlPointTexture = Cuda::CreateTextureObject(bestControlPointCuda, cudaResourceTypeLinear,
                                                             nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, cudaResourceTypeLinear,
                                                          nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);

    const unsigned blocks = (unsigned)CudaContext::GetBlockSize()->reg_updateControlPointPosition;
    const unsigned grids = (unsigned)reg_ceil(sqrtf((float)nVoxels / (float)blocks));
    const dim3 blockDims(blocks, 1, 1);
    const dim3 gridDims(grids, grids, 1);
    reg_updateControlPointPosition_kernel<<<gridDims, blockDims>>>(controlPointImageCuda, *bestControlPointTexture, *gradientImageTexture, (unsigned)nVoxels, scale, optimiseX, optimiseY, optimiseZ);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
