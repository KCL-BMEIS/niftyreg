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

    // The following code is from cutGetMaxGflopsDeviceId()
    int maxGflopsDevice = 0;
    int maxGflops = 0;
    unsigned currentDevice = 0;
    while (currentDevice < numDevices) {
        cudaGetDeviceProperties(&deviceProp, currentDevice);
        int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;
        if (gflops > maxGflops) {
            maxGflops = gflops;
            maxGflopsDevice = currentDevice;
        }
        ++currentDevice;
    }
    NR_CUDA_SAFE_CALL(cudaSetDevice(maxGflopsDevice));
    NR_CUDA_SAFE_CALL(cuCtxCreate(&cudaContext, CU_CTX_SCHED_SPIN, maxGflopsDevice));
    NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, maxGflopsDevice));

    if (deviceProp.major < 1) {
        NR_FATAL_ERROR("The specified graphics card does not exist");
    } else {
        size_t free = 0;
        size_t total = 0;
        cuMemGetInfo(&free, &total);
        if (deviceProp.totalGlobalMem != total)
            NR_FATAL_ERROR("The CUDA card "s + deviceProp.name + " does not seem to be available\n"s +
                           "Expected total memory: "s + std::to_string(deviceProp.totalGlobalMem / (1024 * 1024)) +
                           " MB - Recovered total memory: "s + std::to_string(total / (1024 * 1024)) + " MB"s);
        NR_DEBUG("The following device is used: " << deviceProp.name);
        NR_DEBUG("It has " << free / (1024 * 1024) << " MB free out of " << total / (1024 * 1024) << " MB");
        NR_DEBUG("The CUDA compute capability is " << deviceProp.major << "." << deviceProp.minor);
        NR_DEBUG("The shared memory size in bytes: " << deviceProp.sharedMemPerBlock);
        NR_DEBUG("The CUDA version is " << CUDART_VERSION);
        NR_DEBUG("The card clock rate is " << deviceProp.clockRate / 1000 << " MHz");
        NR_DEBUG("The card has " << deviceProp.multiProcessorCount << " multiprocessors");
        cudaIdx = maxGflopsDevice;
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
