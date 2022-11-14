#include <iostream>
#include "_reg_common_cuda.h"
#include "_reg_tools.h"

void showCUDAInfo(void) {
    // The CUDA card is setup
    cuInit(0);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("-----------------------------------\n");
    printf("[NiftyReg CUDA] %i device(s) detected\n", device_count);
    printf("-----------------------------------\n");

    CUcontext cucontext;

    struct cudaDeviceProp deviceProp;
    // following code is from cutGetMaxGflopsDeviceId()
    int current_device = 0;
    while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);
        if (deviceProp.major > 0) {

            NR_CUDA_SAFE_CALL(cudaSetDevice(current_device));
            NR_CUDA_SAFE_CALL(cuCtxCreate(&cucontext, CU_CTX_SCHED_SPIN, current_device));

            printf("[NiftyReg CUDA] Device id [%i]\n", current_device);
            printf("[NiftyReg CUDA] Device name: %s\n", deviceProp.name);
            size_t free = 0;
            size_t total = 0;
            cuMemGetInfo(&free, &total);
            printf("[NiftyReg CUDA] It has %lu Mb free out of %lu Mb\n",
                   (unsigned long int)(free / (1024 * 1024)),
                   (unsigned long int)(total / (1024 * 1024)));
            printf("[NiftyReg CUDA] Card compute capability: %i.%i\n",
                   deviceProp.major,
                   deviceProp.minor);
            printf("[NiftyReg CUDA] Shared memory size in bytes: %zu\n",
                   deviceProp.sharedMemPerBlock);
            printf("[NiftyReg CUDA] CUDA version %i\n",
                   CUDART_VERSION);
            printf("[NiftyReg CUDA] Card clock rate (Mhz): %i\n",
                   deviceProp.clockRate / 1000);
            printf("[NiftyReg CUDA] Card has %i multiprocessor(s)\n",
                   deviceProp.multiProcessorCount);
        }
        cuCtxDestroy(cucontext);
        ++current_device;
        printf("-----------------------------------\n");
    }
}
