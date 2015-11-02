#include "CUDAContextSingletton.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "_reg_maths.h"
#include "_reg_common_cuda.h"

/* *************************************************************** */
CUDAContextSingletton::CUDAContextSingletton()
{
    // The CUDA card is setup
    cuInit(0);
    int device_count=0;
    cudaGetDeviceCount(&device_count);
#ifndef NDEBUG
        char text[255];
        sprintf(text,"[NiftyReg CUDA] %i card(s) detected\n", device_count);
        reg_print_msg_debug(text);
#endif
    this->cudaContext = NULL;
    this->numDevices = device_count;
    this->cudaIdx = -1;
    pickCard(this->cudaIdx);
}
/* *************************************************************** */
void CUDAContextSingletton::setCudaIdx(int cudaIdxIn)
{
    this->cudaIdx=cudaIdxIn;
    NR_CUDA_SAFE_CALL(cudaSetDevice(this->cudaIdx));
    NR_CUDA_SAFE_CALL(cuCtxCreate(&this->cudaContext, CU_CTX_SCHED_SPIN, this->cudaIdx))
}
/* *************************************************************** */
CUcontext CUDAContextSingletton::getContext()
{
    return this->cudaContext;
}
/* *************************************************************** */
void CUDAContextSingletton::pickCard(unsigned deviceId = -1)
{

    if(deviceId >=0 && deviceId < this->numDevices){
        this->cudaIdx=deviceId;
        //
        NR_CUDA_SAFE_CALL(cudaSetDevice(this->cudaIdx));
        NR_CUDA_SAFE_CALL(cuCtxCreate(&this->cudaContext, CU_CTX_SCHED_SPIN, this->cudaIdx))
        //
       return;
    }

    struct cudaDeviceProp deviceProp;
    // following code is from cutGetMaxGflopsDeviceId()
    int max_gflops_device = 0;
    int max_gflops = 0;
    int current_device = 0;
    while(current_device<this->numDevices ){
        cudaGetDeviceProperties( &deviceProp, current_device );
        int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;
        if( gflops > max_gflops ){
            max_gflops = gflops;
            max_gflops_device = current_device;
        }
        ++current_device;
    }
    NR_CUDA_SAFE_CALL(cudaSetDevice(max_gflops_device));
    NR_CUDA_SAFE_CALL(cuCtxCreate(&this->cudaContext, CU_CTX_SCHED_SPIN, max_gflops_device))
    NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, max_gflops_device));

    if(deviceProp.major<1){
        reg_print_msg_error("[NiftyReg ERROR CUDA] The specified graphical card does not exist.\n");
        reg_exit(1);
    }
    else{
        size_t free=0;
        size_t total=0;
        cuMemGetInfo(&free, &total);
        if(deviceProp.totalGlobalMem != total){
            fprintf(stderr,"[NiftyReg CUDA ERROR] The CUDA card %s does not seem to be available\n",
                   deviceProp.name);
            fprintf(stderr,"[NiftyReg CUDA ERROR] Expected total memory: %lu Mb - Recovered total memory: %lu Mb\n",
                    deviceProp.totalGlobalMem/(1024*1024), total/(1024*1024));
            reg_exit(1);
        }
 #ifndef NDEBUG
            printf("[NiftyReg CUDA] The following device is used: %s\n",
                   deviceProp.name);
            printf("[NiftyReg CUDA] It has %lu Mb free out of %lu Mb\n",
                   (unsigned long int)(free/(1024*1024)),
                   (unsigned long int)(total/(1024*1024)));
            printf("[NiftyReg CUDA] Card compute capability: %i.%i\n",
                   deviceProp.major,
                   deviceProp.minor);
            printf("[NiftyReg CUDA] Shared memory size in bytes: %lu\n",
                   deviceProp.sharedMemPerBlock);
            printf("[NiftyReg CUDA] CUDA version %i\n",
                   CUDART_VERSION);
            printf("[NiftyReg CUDA] Card clock rate: %i MHz\n",
                   deviceProp.clockRate/1000);
            printf("[NiftyReg CUDA] Card has %i multiprocessor(s)\n",
                   deviceProp.multiProcessorCount);
#endif
        this->cudaIdx = max_gflops_device;
    }

}
/* *************************************************************** */
