#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <algorithm>

int main() {

    int deviceCount = 0;
    int output = 0;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);


    // Error when running cudaGetDeviceCount
    if(cudaResultCode != cudaSuccess){
        fprintf(stderr, "%s (CUDA error Code=%d)\n", cudaGetErrorString(cudaResultCode), (int)cudaResultCode);
        return EXIT_FAILURE;
    }

    // Error when running cudaGetDeviceCount
    if(deviceCount == 0){
        fprintf(stderr, "No device detected\n");
        return EXIT_FAILURE;
    }

    //detects device capability and picks the best
    for( unsigned int i = 0; i < deviceCount; ++i ) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        output = std::max(output, deviceProp.major * 10 + deviceProp.minor);
    }

    //	output for device capability
    printf("%i", output);

    return EXIT_SUCCESS;
}
