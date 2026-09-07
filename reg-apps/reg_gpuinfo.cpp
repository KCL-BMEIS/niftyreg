#include "Maths.hpp"
#include "Platform.h"

#ifdef USE_CUDA_PLUGIN
  #include "CudaPluginInterface.h"
#elif defined(USE_CUDA)
  #include "../reg-lib/cuda/_reg_cudainfo.h"
#endif
#ifdef USE_OPENCL
  #include "../reg-lib/cl/_reg_openclinfo.h"
#endif

/* *************************************************************** */
int main()
{
#if defined(USE_CUDA_PLUGIN)
   // The CUDA implementation lives in the runtime plugin, so this binary keeps no
   // CUDA link dependency and can report cleanly on machines without a GPU stack
   if (CudaPluginInterface *cudaPlugin = loadCudaPlugin())
      cudaPlugin->ShowDeviceInfo();
   else
      NR_WARN("CUDA is unavailable: " << getCudaPluginLoadError());
#elif defined(USE_CUDA)
   showCUDAInfo();
#elif !defined(USE_OPENCL)
   NR_WARN("NiftyReg has not been compiled with CUDA or OpenCL");
   NR_WARN("No GPU device information to display");
#endif
#ifdef USE_OPENCL
   showCLInfo();
#endif

    return EXIT_SUCCESS;
}
/* *************************************************************** */
