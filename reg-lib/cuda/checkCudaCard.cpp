#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <algorithm>

int main() {
    int deviceCount = 0, output = 0;
    const cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

    // Error when running cudaGetDeviceCount
    if (cudaResultCode != cudaSuccess) {
        std::cerr << cudaGetErrorString(cudaResultCode) << " (CUDA Error Code=" << cudaResultCode << ")" << std::endl;
        return EXIT_FAILURE;
    }

    // Error when running cudaGetDeviceCount
    if (deviceCount == 0) {
        std::cerr << "No device detected" << std::endl;
        return EXIT_FAILURE;
    }

    // Detect device capability and picks the best
    for (unsigned i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        output = std::max(output, deviceProp.major * 10 + deviceProp.minor);
    }

    // Output for device capability
    std::cout << output;

    return EXIT_SUCCESS;
}
