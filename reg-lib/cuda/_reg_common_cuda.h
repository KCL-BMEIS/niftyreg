/** @file _reg_common_cuda.h
 * @author Marc Modat
 * @date 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "_reg_tools.h"
#include "CudaContext.hpp"

/* *************************************************************** */
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
struct __attribute__((aligned(4))) float4 {
    float x, y, z, w;
};
#endif
/* *************************************************************** */
namespace NiftyReg::Cuda::Internal {
/* *************************************************************** */
inline void SafeCall(const char *file, const int& line) {
#if CUDART_VERSION >= 3200
	cudaError_t err = cudaPeekAtLastError();
#else
	cudaError_t err = cudaDeviceSynchronize();
#endif
	if (err != cudaSuccess) {
		fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
		reg_exit();
	}
}
/* *************************************************************** */
inline void CheckKernel(const char *file, const int& line, const dim3& grid, const dim3& block) {
#if CUDART_VERSION >= 3200
	cudaDeviceSynchronize();
	cudaError_t err = cudaPeekAtLastError();
#else
	cudaError_t err = cudaDeviceSynchronize();
#endif
	if (err != cudaSuccess) {
		fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
		fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
		reg_exit();
	}
#ifndef NDEBUG
	else {
		printf("[NiftyReg CUDA DEBUG] kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
			cudaGetErrorString(cudaGetLastError()), grid.x, grid.y, grid.z, block.x, block.y, block.z);
	}
#endif
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda::Internal
#define NR_CUDA_SAFE_CALL(call) { call; NiftyReg::Cuda::Internal::SafeCall(__FILE__, __LINE__); }
#define NR_CUDA_CHECK_KERNEL(grid, block) NiftyReg::Cuda::Internal::CheckKernel(__FILE__, __LINE__, grid, block)
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray**, int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray**, cudaArray**, int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, int);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, DataType**, int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, cudaArray*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType*, DataType*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, DataType*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, DataType*, DataType*);
/* *************************************************************** */
extern "C++"
void cudaCommon_free(cudaArray*);
/* *************************************************************** */
extern "C++" template <class DataType>
void cudaCommon_free(DataType*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple(DataType*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple1(DataType*, DataType*, const unsigned);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToCpu(DataType*, DataType*, const unsigned);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferArrayFromCpuToDevice(DataType*, DataType*, const unsigned);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferArrayFromDeviceToCpu(DataType*, DataType*, const unsigned);
/* *************************************************************** */
using UniqueTextureObjectPtr = std::unique_ptr<cudaTextureObject_t, void(*)(cudaTextureObject_t*)>;
/* *************************************************************** */
extern "C++"
UniqueTextureObjectPtr cudaCommon_createTextureObject(const void *devPtr,
													  const cudaResourceType& resType,
													  const bool& normalizedCoordinates = false,
													  const size_t& size = 0,
													  const cudaChannelFormatKind& channelFormat = cudaChannelFormatKindNone,
													  const unsigned& channelCount = 1,
													  const cudaTextureFilterMode& filterMode = cudaFilterModeLinear);
/* *************************************************************** */
