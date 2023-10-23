#include "_reg_maths.h"
#include "Platform.h"

#ifdef USE_CUDA
  #include "../reg-lib/cuda/_reg_cudainfo.h"
#endif
#ifdef USE_OPENCL
  #include "../reg-lib/cl/_reg_openclinfo.h"
#endif

/* *************************************************************** */
int main()
{
#ifdef USE_CUDA
   showCUDAInfo();
#else
#ifndef USE_OPENCL
   NR_WARN("NiftyReg has not been compiled with CUDA or OpenCL");
   NR_WARN("No GPU device information to display");
#endif
#endif
#ifdef USE_OPENCL
   showCLInfo();
#endif

    return EXIT_SUCCESS;
}
/* *************************************************************** */
