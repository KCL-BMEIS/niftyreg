#include "../reg-lib/_reg_gpuinfo.h"
#include "_reg_maths.h"
#include "Platform.h"

void showCPUInfo()
{
    reg_print_msg_error("You asked for CPU info not GPU info");
}
/* *************************************************************** */
int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int platformCode = atoi(argv[1]);
    if (platformCode == NR_PLATFORM_CPU) {
        showCPUInfo();
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        showCUDAInfo();
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        showCLInfo();
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
