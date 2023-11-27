#include "CudaOptimiser.hpp"
#include "_reg_common_cuda_kernels.cu"
#include <curand_kernel.h>

/* *************************************************************** */
namespace NiftyReg {
/* *************************************************************** */
CudaOptimiser::CudaOptimiser(): Optimiser<float>::Optimiser() {
    this->currentDofCuda = nullptr;
    this->currentDofBwCuda = nullptr;
    this->bestDofCuda = nullptr;
    this->bestDofBwCuda = nullptr;
    this->gradientCuda = nullptr;
    this->gradientBwCuda = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
CudaOptimiser::~CudaOptimiser() {
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
void CudaOptimiser::Initialise(size_t nvox,
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
void CudaOptimiser::RestoreBestDof() {
    // Restore forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentDofCuda, this->bestDofCuda, this->GetVoxNumber() * sizeof(float4), cudaMemcpyDeviceToDevice));
    // Restore backward transformation if required
    if (this->isSymmetric)
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentDofBwCuda, this->bestDofBwCuda, this->GetVoxNumberBw() * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void CudaOptimiser::StoreCurrentDof() {
    // Store forward transformation
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestDofCuda, this->currentDofCuda, this->GetVoxNumber() * sizeof(float4), cudaMemcpyDeviceToDevice));
    // Store backward transformation if required
    if (this->isSymmetric)
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestDofBwCuda, this->currentDofBwCuda, this->GetVoxNumberBw() * sizeof(float4), cudaMemcpyDeviceToDevice));
}
/* *************************************************************** */
void CudaOptimiser::Perturbation(float length) {
    // Reset the number of iteration
    this->currentIterationNumber = 0;

    auto perturbate = []__device__(float4 *currentDofCuda, cudaTextureObject_t bestDofTexture, const float length, const size_t index) {
        curandState_t state;
        curand_init(clock64(), index, 0, &state);
        const float4 bestDofVal = tex1Dfetch<float4>(bestDofTexture, index);
        float4 curDofVal = currentDofCuda[index];
        curDofVal.x = bestDofVal.x + length * curand_uniform(&state);
        curDofVal.y = bestDofVal.y + length * curand_uniform(&state);
        curDofVal.z = bestDofVal.z + length * curand_uniform(&state);
        curDofVal.w = bestDofVal.w + length * curand_uniform(&state);
        currentDofCuda[index] = curDofVal;
    };

    // Create some perturbation for degree of freedom
    const size_t voxNumber = this->GetVoxNumber();
    auto currentDofCuda = this->currentDofCuda;
    auto bestDofTexturePtr = Cuda::CreateTextureObject(this->bestDofCuda, voxNumber, cudaChannelFormatKindFloat, 4);
    auto bestDofTexture = *bestDofTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxNumber, [=]__device__(const size_t index) {
        perturbate(currentDofCuda, bestDofTexture, length, index);
    });
    if (this->isSymmetric) {
        const size_t voxNumberBw = this->GetVoxNumberBw();
        auto currentDofBwCuda = this->currentDofBwCuda;
        auto bestDofBwTexturePtr = Cuda::CreateTextureObject(this->bestDofBwCuda, voxNumberBw, cudaChannelFormatKindFloat, 4);
        auto bestDofBwTexture = *bestDofBwTexturePtr;
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxNumberBw, [=]__device__(const size_t index) {
            perturbate(currentDofBwCuda, bestDofBwTexture, length, index);
        });
    }
    this->StoreCurrentDof();
    this->currentObjFunctionValue = this->bestObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();
}
/* *************************************************************** */
CudaConjugateGradient::CudaConjugateGradient(): CudaOptimiser::CudaOptimiser() {
    this->array1 = nullptr;
    this->array1Bw = nullptr;
    this->array2 = nullptr;
    this->array2Bw = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
CudaConjugateGradient::~CudaConjugateGradient() {
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
void CudaConjugateGradient::Initialise(size_t nvox,
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
    CudaOptimiser::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
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
void CudaConjugateGradient::Optimise(float maxLength,
                                         float smallLength,
                                         float& startLength) {
    this->UpdateGradientValues();
    CudaOptimiser::Optimise(maxLength, smallLength, startLength);
}
/* *************************************************************** */
void CudaConjugateGradient::Perturbation(float length) {
    CudaOptimiser::Perturbation(length);
    this->firstCall = true;
}
/* *************************************************************** */
void InitialiseConjugateGradient(float4 *gradientCuda, float4 *conjugateGCuda, float4 *conjugateHCuda, const size_t nVoxels) {
    auto gradientTexturePtr = Cuda::CreateTextureObject(gradientCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto gradientTexture = *gradientTexturePtr;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), nVoxels, [=]__device__(const int index) {
        const float4 gradValue = tex1Dfetch<float4>(gradientTexture, index);
        conjugateGCuda[index] = conjugateHCuda[index] = make_float4(-gradValue.x, -gradValue.y, -gradValue.z, 0);
    });
}
/* *************************************************************** */
void GetConjugateGradient(float4 *gradientCuda,
                          float4 *conjugateGCuda,
                          float4 *conjugateHCuda,
                          const size_t nVoxels,
                          const bool isSymmetric,
                          float4 *gradientBwCuda,
                          float4 *conjugateGBwCuda,
                          float4 *conjugateHBwCuda,
                          const size_t nVoxelsBw) {
    auto gradientTexturePtr = Cuda::CreateTextureObject(gradientCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto conjugateGTexturePtr = Cuda::CreateTextureObject(conjugateGCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto conjugateHTexturePtr = Cuda::CreateTextureObject(conjugateHCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto gradientTexture = *gradientTexturePtr;
    auto conjugateGTexture = *conjugateGTexturePtr;
    auto conjugateHTexture = *conjugateHTexturePtr;
    Cuda::UniqueTextureObjectPtr gradientBwTexturePtr, conjugateGBwTexturePtr, conjugateHBwTexturePtr;
    cudaTextureObject_t gradientBwTexture = 0, conjugateGBwTexture = 0, conjugateHBwTexture = 0;
    if (isSymmetric) {
        gradientBwTexturePtr = Cuda::CreateTextureObject(gradientBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
        conjugateGBwTexturePtr = Cuda::CreateTextureObject(conjugateGBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
        conjugateHBwTexturePtr = Cuda::CreateTextureObject(conjugateHBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
        gradientBwTexture = *gradientBwTexturePtr;
        conjugateGBwTexture = *conjugateGBwTexturePtr;
        conjugateHBwTexture = *conjugateHBwTexturePtr;
    }

    // gam = sum((grad+g)*grad)/sum(HxG);
    auto calcGam = []__device__(cudaTextureObject_t gradientTexture, cudaTextureObject_t conjugateGTexture,
                                cudaTextureObject_t conjugateHTexture, const int index) {
        const float4 hValue = tex1Dfetch<float4>(conjugateHTexture, index);
        const float4 gValue = tex1Dfetch<float4>(conjugateGTexture, index);
        const float gg = gValue.x * hValue.x + gValue.y * hValue.y + gValue.z * hValue.z;

        const float4 grad = tex1Dfetch<float4>(gradientTexture, index);
        const float dgg = (grad.x + gValue.x) * grad.x + (grad.y + gValue.y) * grad.y + (grad.z + gValue.z) * grad.z;

        return make_double2(dgg, gg);
    };

    float gam;
    thrust::counting_iterator<int> it(0);
    const double2 gg = thrust::transform_reduce(thrust::device, it, it + nVoxels, [=]__device__(const int index) {
        return calcGam(gradientTexture, conjugateGTexture, conjugateHTexture, index);
    }, make_double2(0, 0), thrust::plus<double2>());
    if (isSymmetric) {
        it = thrust::counting_iterator<int>(0);
        const double2 ggBw = thrust::transform_reduce(thrust::device, it, it + nVoxelsBw, [=]__device__(const int index) {
            return calcGam(gradientBwTexture, conjugateGBwTexture, conjugateHBwTexture, index);
        }, make_double2(0, 0), thrust::plus<double2>());
        gam = static_cast<float>((gg.x + ggBw.x) / (gg.y + ggBw.y));
    } else gam = static_cast<float>(gg.x / gg.y);

    // Conjugate gradient
    auto conjugate = [gam]__device__(float4 *gradientCuda, float4 *conjugateGCuda, float4 *conjugateHCuda,
                                     cudaTextureObject_t gradientTexture, cudaTextureObject_t conjugateHTexture, const int index) {
        // G = -grad
        float4 gradGValue = tex1Dfetch<float4>(gradientTexture, index);
        gradGValue = make_float4(-gradGValue.x, -gradGValue.y, -gradGValue.z, 0);
        conjugateGCuda[index] = gradGValue;

        // H = G + gam * H
        float4 gradHValue = tex1Dfetch<float4>(conjugateHTexture, index);
        gradHValue = make_float4(gradGValue.x + gam * gradHValue.x,
                                 gradGValue.y + gam * gradHValue.y,
                                 gradGValue.z + gam * gradHValue.z, 0);
        conjugateHCuda[index] = gradHValue;

        gradientCuda[index] = make_float4(-gradHValue.x, -gradHValue.y, -gradHValue.z, 0);
    };

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), nVoxels, [=]__device__(const int index) {
        conjugate(gradientCuda, conjugateGCuda, conjugateHCuda, gradientTexture, conjugateHTexture, index);
    });
    if (isSymmetric) {
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), nVoxelsBw, [=]__device__(const int index) {
            conjugate(gradientBwCuda, conjugateGBwCuda, conjugateHBwCuda, gradientBwTexture, conjugateHBwTexture, index);
        });
    }
}
/* *************************************************************** */
void CudaConjugateGradient::UpdateGradientValues() {
    if (this->firstCall) {
        NR_DEBUG("Conjugate gradient initialisation");
        InitialiseConjugateGradient(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber());
        if (this->isSymmetric)
            InitialiseConjugateGradient(this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
        this->firstCall = false;
    } else {
        NR_DEBUG("Conjugate gradient update");
        GetConjugateGradient(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber(),
                             this->isSymmetric, this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
    }
}
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
