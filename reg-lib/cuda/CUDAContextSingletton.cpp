#include "CUDAContextSingletton.h"
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
   this->cudaIdx = 999;
   pickCard(this->cudaIdx);
}
/* *************************************************************** */
void CUDAContextSingletton::setCudaIdx(unsigned int cudaIdxIn)
{
   if (cudaIdxIn>=this->numDevices){
      reg_print_msg_error("The specified cuda card id is not defined");
      reg_print_msg_error("Run reg_gpuinfo to get the proper id");
      reg_exit();
   }
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
void CUDAContextSingletton::pickCard(unsigned deviceId = 999)
{
    struct cudaDeviceProp deviceProp;
    if(deviceId < this->numDevices) {
      this->cudaIdx=deviceId;
      //
      NR_CUDA_SAFE_CALL(cudaSetDevice(this->cudaIdx));
      NR_CUDA_SAFE_CALL(cuCtxCreate(&this->cudaContext, CU_CTX_SCHED_SPIN, this->cudaIdx));
      //
      cudaGetDeviceProperties(&deviceProp, this->cudaIdx);
      if(deviceProp.major > 1) {
          this->isCardDoubleCapable = true;
      }
      else if(deviceProp.major == 1 && deviceProp.minor > 2) {
          this->isCardDoubleCapable = true;
      } else {
          this->isCardDoubleCapable = false;
      }
      //
      return;
    }

   // following code is from cutGetMaxGflopsDeviceId()
   int max_gflops_device = 0;
   int max_gflops = 0;
   unsigned int current_device = 0;
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
      reg_exit();
   }
   else{
      size_t free=0;
      size_t total=0;
      cuMemGetInfo(&free, &total);
      if(deviceProp.totalGlobalMem != total){
         fprintf(stderr,"[NiftyReg CUDA ERROR] The CUDA card %s does not seem to be available\n",
                 deviceProp.name);
         fprintf(stderr,"[NiftyReg CUDA ERROR] Expected total memory: %zu Mb - Recovered total memory: %zu Mb\n",
                 deviceProp.totalGlobalMem/(1024*1024), total/(1024*1024));
         reg_exit();
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
      printf("[NiftyReg CUDA] Shared memory size in bytes: %zu\n",
             deviceProp.sharedMemPerBlock);
      printf("[NiftyReg CUDA] CUDA version %i\n",
             CUDART_VERSION);
      printf("[NiftyReg CUDA] Card clock rate: %i MHz\n",
             deviceProp.clockRate/1000);
      printf("[NiftyReg CUDA] Card has %i multiprocessor(s)\n",
             deviceProp.multiProcessorCount);
#endif
      this->cudaIdx = max_gflops_device;
      //
      cudaGetDeviceProperties(&deviceProp, this->cudaIdx);
      if(deviceProp.major > 1) {
          this->isCardDoubleCapable = true;
      }
      else if(deviceProp.major == 1 && deviceProp.minor > 2) {
          this->isCardDoubleCapable = true;
      } else {
          this->isCardDoubleCapable = false;
      }
      //
   }
}
/* *************************************************************** */
bool CUDAContextSingletton::getIsCardDoubleCapable()
{
    return this->isCardDoubleCapable;
}
/* *************************************************************** */
CUDAContextSingletton::~CUDAContextSingletton()
{
   cuCtxDestroy(this->cudaContext);
}
