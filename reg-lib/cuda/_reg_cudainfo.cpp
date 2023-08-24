#include "CudaCommon.hpp"
#include "_reg_tools.h"

void showCUDAInfo() {
    // The CUDA card is setup
    cuInit(0);

    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    NR_COUT << "-----------------------------------" << std::endl;
    NR_COUT << "[NiftyReg CUDA] " << numDevices << " device(s) detected" << std::endl;
    NR_COUT << "-----------------------------------" << std::endl;

    CUcontext cuContext;
    struct cudaDeviceProp deviceProp;
    // following code is from cutGetMaxGflopsDeviceId()
    int currentDevice = 0;
    while (currentDevice < numDevices) {
        cudaGetDeviceProperties(&deviceProp, currentDevice);
        if (deviceProp.major > 0) {
            NR_CUDA_SAFE_CALL(cudaSetDevice(currentDevice));
            NR_CUDA_SAFE_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN, currentDevice));
            NR_COUT << "[NiftyReg CUDA] Device ID: " << currentDevice << std::endl;
            NR_COUT << "[NiftyReg CUDA] Device name: " << deviceProp.name << std::endl;
            size_t free = 0, total = 0;
            cuMemGetInfo(&free, &total);
            NR_COUT << "[NiftyReg CUDA] It has " << free / (1024 * 1024) << " MB free out of " << total / (1024 * 1024) << " MB" << std::endl;
            NR_COUT << "[NiftyReg CUDA] Card compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            NR_COUT << "[NiftyReg CUDA] Shared memory size in bytes: " << deviceProp.sharedMemPerBlock << std::endl;
            NR_COUT << "[NiftyReg CUDA] CUDA version " << CUDART_VERSION << std::endl;
            NR_COUT << "[NiftyReg CUDA] Card clock rate (Mhz): " << deviceProp.clockRate / 1000 << std::endl;
            NR_COUT << "[NiftyReg CUDA] Card has " << deviceProp.multiProcessorCount << " multiprocessor(s)" << std::endl;
        }
        cuCtxDestroy(cuContext);
        ++currentDevice;
        NR_COUT << "-----------------------------------" << std::endl;
    }
}
