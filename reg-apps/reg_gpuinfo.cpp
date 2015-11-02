#include "_reg_maths.h"
#include "Platform.h"

#ifdef _USE_CUDA
  #include "../reg-lib/cuda/_reg_cudainfo.h"
#endif
#ifdef _USE_OPENCL
  #include "../reg-lib/cl/_reg_openclinfo.h"
#endif

/* *************************************************************** */
int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int platformCode = atoi(argv[1]);
    if (platformCode == NR_PLATFORM_CUDA) {
#ifdef _USE_CUDA
        showCUDAInfo();
#else
        reg_print_msg_error("NiftyReg has not been compiled with CUDA");
        return EXIT_FAILURE;
#endif
    }
    else if (platformCode == NR_PLATFORM_CL) {
#ifdef _USE_OPENCL
        showCLInfo();
#else
        reg_print_msg_error("NiftyReg has not been compiled with OpenCL");
        return EXIT_FAILURE;
#endif
    }
    else {
       reg_print_msg_error("The platform code is not supported");
       reg_print_msg_error("Expected value(s):");
#ifdef _USE_CUDA
       reg_print_msg_error("1 - CUDA");
#endif
#ifdef _USE_OPENCL
       reg_print_msg_error("2 - OpenCL");
#endif
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
