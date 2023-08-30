#include "CudaContext.hpp"
#include "CudaCommon.hpp"

namespace NiftyReg {
/* *************************************************************** */
CudaContext::CudaContext() {
    // The CUDA card is setup
    cuInit(0);
    numDevices = 0;
    cudaGetDeviceCount((int*)&numDevices);
    NR_DEBUG(numDevices << " CUDA card(s) detected");
    cudaContext = nullptr;
    cudaIdx = 999;
    PickCard(cudaIdx);
}
/* *************************************************************** */
void CudaContext::SetCudaIdx(unsigned cudaIdxIn) {
    if (cudaIdxIn >= numDevices)
        NR_FATAL_ERROR("The specified CUDA card ID is not defined! Run reg_gpuinfo to get the proper id.");
    cudaIdx = cudaIdxIn;
    PickCard(cudaIdx);
}
/* *************************************************************** */
CUcontext CudaContext::GetContext() {
    return cudaContext;
}
/* *************************************************************** */
void CudaContext::SetBlockSize(int major) {
    if (major >= 3)
        blockSize.reset(new BlockSize300());
    else
        blockSize.reset(new BlockSize100());
}
/* *************************************************************** */
void CudaContext::PickCard(unsigned deviceId = 999) {
    struct cudaDeviceProp deviceProp;
    if (deviceId < numDevices) {
        cudaIdx = deviceId;
        NR_CUDA_SAFE_CALL(cudaSetDevice(cudaIdx));
        NR_CUDA_SAFE_CALL(cuCtxCreate(&cudaContext, CU_CTX_SCHED_SPIN, cudaIdx));

        cudaGetDeviceProperties(&deviceProp, cudaIdx);
        if (deviceProp.major > 1) {
            isCardDoubleCapable = true;
        } else if (deviceProp.major == 1 && deviceProp.minor > 2) {
            isCardDoubleCapable = true;
        } else {
            isCardDoubleCapable = false;
        }
        SetBlockSize(deviceProp.major);
        return;
    }

    // following code is from cutGetMaxGflopsDeviceId()
    int max_gflops_device = 0;
    int max_gflops = 0;
    unsigned current_device = 0;
    while (current_device < numDevices) {
        cudaGetDeviceProperties(&deviceProp, current_device);
        int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;
        if (gflops > max_gflops) {
            max_gflops = gflops;
            max_gflops_device = current_device;
        }
        ++current_device;
    }
    NR_CUDA_SAFE_CALL(cudaSetDevice(max_gflops_device));
    NR_CUDA_SAFE_CALL(cuCtxCreate(&cudaContext, CU_CTX_SCHED_SPIN, max_gflops_device));
    NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, max_gflops_device));

    if (deviceProp.major < 1) {
        NR_FATAL_ERROR("The specified graphics card does not exist");
    } else {
        size_t free = 0;
        size_t total = 0;
        cuMemGetInfo(&free, &total);
        if (deviceProp.totalGlobalMem != total)
            NR_FATAL_ERROR("The CUDA card "s + deviceProp.name + " does not seem to be available\n"s +
                           "Expected total memory: "s + std::to_string(deviceProp.totalGlobalMem / (1024 * 1024)) +
                           " MB - Recovered total memory: "s + std::to_string(total / (1024 * 1024)) + " MB");
        NR_DEBUG("The following device is used: "s + deviceProp.name);
        NR_DEBUG("It has "s + std::to_string(free / (1024 * 1024)) + " MB free out of "s + std::to_string(total / (1024 * 1024)) + " MB");
        NR_DEBUG("The CUDA compute capability is "s + std::to_string(deviceProp.major) + "."s + std::to_string(deviceProp.minor));
        NR_DEBUG("The shared memory size in bytes: "s + std::to_string(deviceProp.sharedMemPerBlock));
        NR_DEBUG("The CUDA version is "s + std::to_string(CUDART_VERSION));
        NR_DEBUG("The card clock rate is "s + std::to_string(deviceProp.clockRate / 1000) + " MHz");
        NR_DEBUG("The card has "s + std::to_string(deviceProp.multiProcessorCount) + " multiprocessors");
        cudaIdx = max_gflops_device;
        cudaGetDeviceProperties(&deviceProp, cudaIdx);
        if (deviceProp.major > 1) {
            isCardDoubleCapable = true;
        } else if (deviceProp.major == 1 && deviceProp.minor > 2) {
            isCardDoubleCapable = true;
        } else {
            isCardDoubleCapable = false;
        }
        SetBlockSize(deviceProp.major);
    }
}
/* *************************************************************** */
bool CudaContext::IsCardDoubleCapable() {
    return isCardDoubleCapable;
}
/* *************************************************************** */
CudaContext::~CudaContext() {
    cuCtxDestroy(cudaContext);
}
/* *************************************************************** */
} // namespace NiftyReg
