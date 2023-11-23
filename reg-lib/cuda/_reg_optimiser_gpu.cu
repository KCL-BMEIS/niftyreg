#include "_reg_optimiser_gpu.h"
#include "_reg_optimiser_kernels.cu"
#include "_reg_common_cuda_kernels.cu"
#include <curand_kernel.h>

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
        NR_DEBUG("Conjugate gradient initialisation");
        reg_initialiseConjugateGradient_gpu(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber());
        if (this->isSymmetric)
            reg_initialiseConjugateGradient_gpu(this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
        this->firstCall = false;
    } else {
        NR_DEBUG("Conjugate gradient update");
        reg_getConjugateGradient_gpu(this->gradientCuda, this->array1, this->array2, this->GetVoxNumber(),
                                     this->isSymmetric, this->gradientBwCuda, this->array1Bw, this->array2Bw, this->GetVoxNumberBw());
    }
}
/* *************************************************************** */
void reg_conjugateGradient_gpu::Optimise(float maxLength,
                                         float smallLength,
                                         float& startLength) {
    this->UpdateGradientValues();
    reg_optimiser_gpu::Optimise(maxLength, smallLength, startLength);
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
                                         const size_t nVoxels) {
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, nVoxels, cudaChannelFormatKindFloat, 4);

    const unsigned blocks = CudaContext::GetBlockSize()->reg_initialiseConjugateGradient;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)nVoxels / (float)blocks));
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
                                  const size_t nVoxels,
                                  const bool isSymmetric,
                                  float4 *gradientImageBwCuda,
                                  float4 *conjugateGBwCuda,
                                  float4 *conjugateHBwCuda,
                                  const size_t nVoxelsBw) {
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto conjugateGTexture = Cuda::CreateTextureObject(conjugateGCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto conjugateHTexture = Cuda::CreateTextureObject(conjugateHCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    Cuda::UniqueTextureObjectPtr gradientImageBwTexture, conjugateGBwTexture, conjugateHBwTexture;
    if (isSymmetric) {
        gradientImageBwTexture = Cuda::CreateTextureObject(gradientImageBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
        conjugateGBwTexture = Cuda::CreateTextureObject(conjugateGBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
        conjugateHBwTexture = Cuda::CreateTextureObject(conjugateHBwCuda, nVoxelsBw, cudaChannelFormatKindFloat, 4);
    }

    // gam = sum((grad+g)*grad)/sum(HxG);
    unsigned blocks = CudaContext::GetBlockSize()->reg_getConjugateGradient1;
    unsigned grids = (unsigned)Ceil(sqrtf((float)nVoxels / (float)blocks));
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
        grids = (unsigned)Ceil(sqrtf((float)nVoxelsBw / (float)blocks));
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
    grids = (unsigned)Ceil(sqrtf((float)nVoxels / (float)blocks));
    gridDims = dim3(blocks, 1, 1);
    blockDims = dim3(grids, grids, 1);
    reg_getConjugateGradient2_kernel<<<blockDims, gridDims>>>(gradientImageCuda, conjugateGCuda, conjugateHCuda, (unsigned)nVoxels, gam);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    if (isSymmetric) {
        grids = (unsigned)Ceil(sqrtf((float)nVoxelsBw / (float)blocks));
        gridDims = dim3(blocks, 1, 1);
        blockDims = dim3(grids, grids, 1);
        reg_getConjugateGradient2_kernel<<<blockDims, gridDims>>>(gradientImageBwCuda, conjugateGBwCuda, conjugateHBwCuda, (unsigned)nVoxelsBw, gam);
        NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
    }
}
/* *************************************************************** */
void reg_updateControlPointPosition_gpu(const size_t nVoxels,
                                        float4 *controlPointImageCuda,
                                        const float4 *bestControlPointCuda,
                                        const float4 *gradientImageCuda,
                                        const float scale,
                                        const bool optimiseX,
                                        const bool optimiseY,
                                        const bool optimiseZ) {
    auto bestControlPointTexture = Cuda::CreateTextureObject(bestControlPointCuda, nVoxels, cudaChannelFormatKindFloat, 4);
    auto gradientImageTexture = Cuda::CreateTextureObject(gradientImageCuda, nVoxels, cudaChannelFormatKindFloat, 4);

    const unsigned blocks = (unsigned)CudaContext::GetBlockSize()->reg_updateControlPointPosition;
    const unsigned grids = (unsigned)Ceil(sqrtf((float)nVoxels / (float)blocks));
    const dim3 blockDims(blocks, 1, 1);
    const dim3 gridDims(grids, grids, 1);
    reg_updateControlPointPosition_kernel<<<gridDims, blockDims>>>(controlPointImageCuda, *bestControlPointTexture, *gradientImageTexture,
                                                                   (unsigned)nVoxels, scale, optimiseX, optimiseY, optimiseZ);
    NR_CUDA_CHECK_KERNEL(gridDims, blockDims);
}
/* *************************************************************** */
