#include "CudaNormaliseGradient.hpp"
#include "CudaTools.hpp"

/* *************************************************************** */
template<bool optimiseX, bool optimiseY, bool optimiseZ>
float GetMaximalLength(const float4 *imageCuda, const size_t nVoxels) {
    auto imageTexturePtr = Cuda::CreateTextureObject(imageCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto imageTexture = *imageTexturePtr;
    thrust::counting_iterator<unsigned> index(0);
    return thrust::transform_reduce(thrust::device, index, index + nVoxels, [=]__device__(const unsigned index) {
        const float4 val = tex1Dfetch<float4>(imageTexture, index);
        return sqrtf((optimiseX ? Square(val.x) : 0) +
                     (optimiseY ? Square(val.y) : 0) +
                     (optimiseZ ? Square(val.z) : 0));
    }, 0.f, thrust::maximum<float>());
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY>
static inline float GetMaximalLength(const float4 *imageCuda,
                                     const size_t nVoxels,
                                     const bool optimiseZ) {
    auto getMaximalLength = GetMaximalLength<optimiseX, optimiseY, true>;
    if (!optimiseZ) getMaximalLength = GetMaximalLength<optimiseX, optimiseY, false>;
    return getMaximalLength(imageCuda, nVoxels);
}
/* *************************************************************** */
template<bool optimiseX>
static inline float GetMaximalLength(const float4 *imageCuda,
                                     const size_t nVoxels,
                                     const bool optimiseY,
                                     const bool optimiseZ) {
    auto getMaximalLength = GetMaximalLength<optimiseX, true>;
    if (!optimiseY) getMaximalLength = GetMaximalLength<optimiseX, false>;
    return getMaximalLength(imageCuda, nVoxels, optimiseZ);
}
/* *************************************************************** */
float NiftyReg::Cuda::GetMaximalLength(const float4 *imageCuda,
                                       const size_t nVoxels,
                                       const bool optimiseX,
                                       const bool optimiseY,
                                       const bool optimiseZ) {
    auto getMaximalLength = ::GetMaximalLength<true>;
    if (!optimiseX) getMaximalLength = ::GetMaximalLength<false>;
    return getMaximalLength(imageCuda, nVoxels, optimiseY, optimiseZ);
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY, bool optimiseZ>
void NormaliseGradient(float4 *imageCuda, const size_t nVoxels, const double maxGradLengthInv) {
    auto imageTexturePtr = Cuda::CreateTextureObject(imageCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto imageTexture = *imageTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned>(0), nVoxels, [=]__device__(const unsigned index) {
        const float4 val = tex1Dfetch<float4>(imageTexture, index);
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
