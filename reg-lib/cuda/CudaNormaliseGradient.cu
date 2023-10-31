#include "CudaNormaliseGradient.hpp"
#include "_reg_tools_gpu.h"

/* *************************************************************** */
__global__ static void GetMaximalLengthKernel(float *dists,
                                              cudaTextureObject_t imageTexture,
                                              const unsigned nVoxels,
                                              const bool optimiseX,
                                              const bool optimiseY,
                                              const bool optimiseZ) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nVoxels) {
        float4 gradValue = tex1Dfetch<float4>(imageTexture, tid);
        dists[tid] = sqrtf((optimiseX ? Square(gradValue.x) : 0) +
                           (optimiseY ? Square(gradValue.y) : 0) +
                           (optimiseZ ? Square(gradValue.z) : 0));
    }
}
/* *************************************************************** */
float NiftyReg::Cuda::GetMaximalLength(const float4 *imageCuda,
                                       const size_t nVoxels,
                                       const bool optimiseX,
                                       const bool optimiseY,
                                       const bool optimiseZ) {
    // Create a texture object for the imageCuda
    auto imageTexture = Cuda::CreateTextureObject(imageCuda, cudaResourceTypeLinear,
                                                  nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);

    float *dists = nullptr;
    NR_CUDA_SAFE_CALL(cudaMalloc(&dists, nVoxels * sizeof(float)));

    const unsigned threads = CudaContext::GetBlockSize()->GetMaximalLength;
    const unsigned blocks = static_cast<unsigned>(Ceil(sqrtf(static_cast<float>(nVoxels) / static_cast<float>(threads))));
    dim3 blockDims(threads, 1, 1);
    dim3 gridDims(blocks, blocks, 1);
    GetMaximalLengthKernel<<<gridDims, blockDims>>>(dists, *imageTexture, static_cast<unsigned>(nVoxels), optimiseX, optimiseY, optimiseZ);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);

    const float maxDistance = reg_maxReduction_gpu(dists, nVoxels);
    NR_CUDA_SAFE_CALL(cudaFree(dists));

    return maxDistance;
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY, bool optimiseZ>
void NormaliseGradient(float4 *imageCuda, const size_t nVoxels, const double maxGradLengthInv) {
    auto imageTexturePtr = Cuda::CreateTextureObject(imageCuda, cudaResourceTypeLinear,
                                                     nVoxels * sizeof(float4), cudaChannelFormatKindFloat, 4);
    auto imageTexture = *imageTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned>(0), nVoxels, [=]__device__(const unsigned index) {
        const float4& val = tex1Dfetch<float4>(imageTexture, index);
        imageCuda[index] = make_float4(optimiseX ? val.x * maxGradLengthInv : 0,
                                       optimiseY ? val.y * maxGradLengthInv : 0,
                                       optimiseZ ? val.z * maxGradLengthInv : 0,
                                       val.w);
    });
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY>
static inline void NormaliseGradient(float4 *imageCuda,
                                     const size_t nVoxels,
                                     const double maxGradLengthInv,
                                     const bool optimiseZ) {
    auto normaliseGradient = NormaliseGradient<optimiseX, optimiseY, true>;
    if (!optimiseZ) normaliseGradient = NormaliseGradient<optimiseX, optimiseY, false>;
    normaliseGradient(imageCuda, nVoxels, maxGradLengthInv);
}
/* *************************************************************** */
template<bool optimiseX>
static inline void NormaliseGradient(float4 *imageCuda,
                                     const size_t nVoxels,
                                     const double maxGradLengthInv,
                                     const bool optimiseY,
                                     const bool optimiseZ) {
    auto normaliseGradient = NormaliseGradient<optimiseX, true>;
    if (!optimiseY) normaliseGradient = NormaliseGradient<optimiseX, false>;
    normaliseGradient(imageCuda, nVoxels, maxGradLengthInv, optimiseZ);
}
/* *************************************************************** */
void NiftyReg::Cuda::NormaliseGradient(float4 *imageCuda,
                                       const size_t nVoxels,
                                       const double maxGradLength,
                                       const bool optimiseX,
                                       const bool optimiseY,
                                       const bool optimiseZ) {
    auto normaliseGradient = ::NormaliseGradient<true>;
    if (!optimiseX) normaliseGradient = ::NormaliseGradient<false>;
    normaliseGradient(imageCuda, nVoxels, 1.0 / maxGradLength, optimiseY, optimiseZ);
}
/* *************************************************************** */
