#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include <algorithm>

int main() {

	int deviceCount = 0;
	int output = 0;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

	// Error when running cudaGetDeviceCount
	if( cudaResultCode != cudaSuccess || deviceCount == 0 ) // cudaSuccess=0
		return EXIT_FAILURE;

	//detects device capability and picks the lowest
	for( unsigned int i = 0; i < deviceCount; ++i ) {
		cudaSetDevice(i);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		output = std::max(output, deviceProp.major * 10 + deviceProp.minor);
	}

	//	output for device capability
	printf("%1.1f", output / 10.0);

	return EXIT_SUCCESS;
}
