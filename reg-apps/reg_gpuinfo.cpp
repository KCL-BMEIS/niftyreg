#include "_reg_maths.h"
#include "Platform.h"

#ifdef _USE_CUDA
  #include "../reg-lib/cuda/_reg_cudainfo.h"
#endif
#ifdef _USE_OPENCL
  #include "../reg-lib/cl/_reg_openclinfo.h"
#endif

/* *************************************************************** */
int main()
{
#ifdef _USE_CUDA
   showCUDAInfo();
#ifndef _USE_OPENCL
   reg_print_msg_warn("NiftyReg has not been compiled with CUDA or OpenCL");
   reg_print_msg_warn("No GPU device information to display");
#endif
#endif
#ifdef _USE_OPENCL
   showCLInfo();
#endif

    return EXIT_SUCCESS;
}
/* *************************************************************** */
