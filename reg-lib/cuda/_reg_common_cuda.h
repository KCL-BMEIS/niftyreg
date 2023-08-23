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
inline void SafeCall(const std::string& file, const int& line, const std::string& funcName) {
#if CUDART_VERSION >= 3200
	const cudaError_t err = cudaPeekAtLastError();
#else
	const cudaError_t err = cudaDeviceSynchronize();
#endif
	if (err != cudaSuccess)
        NiftyReg::Internal::FatalError(file, line, funcName, "CUDA error: "s + cudaGetErrorString(err));
}
/* *************************************************************** */
inline void CheckKernel(const std::string& file, const int& line, const std::string& funcName, const dim3& grid, const dim3& block) {
#if CUDART_VERSION >= 3200
	cudaDeviceSynchronize();
	const cudaError_t err = cudaPeekAtLastError();
#else
	const cudaError_t err = cudaDeviceSynchronize();
#endif
	if (err != cudaSuccess) {
        NiftyReg::Internal::FatalError(file, line, funcName, "CUDA error: "s + cudaGetErrorString(err) +
                "\n\tGrid size ["s + std::to_string(grid.x) + " "s + std::to_string(grid.y) + " "s + std::to_string(grid.z) +
                "] - Block size ["s + std::to_string(block.x) + " "s + std::to_string(block.y) + " "s + std::to_string(block.z) + "]");
	} else {
        NR_DEBUG("CUDA kernel: "s + cudaGetErrorString(err) +
                 " - Grid size ["s + std::to_string(grid.x) + " "s + std::to_string(grid.y) + " "s + std::to_string(grid.z) +
                 "] - Block size ["s + std::to_string(block.x) + " "s + std::to_string(block.y) + " "s + std::to_string(block.z) + "]");
	}
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda::Internal
#define NR_CUDA_SAFE_CALL(call)             { call; NiftyReg::Cuda::Internal::SafeCall(__FILE__, __LINE__, NR_FUNCTION); }
#define NR_CUDA_CHECK_KERNEL(grid, block)   NiftyReg::Cuda::Internal::CheckKernel(__FILE__, __LINE__, NR_FUNCTION, grid, block)
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray**, const int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray**, cudaArray**, const int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, const size_t&);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, const int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType**, DataType**, const int*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, const nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, cudaArray*, const nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType*, const nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType*, DataType*, const nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, const DataType*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, const DataType*, const DataType*);
/* *************************************************************** */
extern "C++"
void cudaCommon_free(cudaArray*);
/* *************************************************************** */
extern "C++" template <class DataType>
void cudaCommon_free(DataType*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple(DataType*, const nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple1(DataType*, const DataType*, const size_t&);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferFromDeviceToCpu(DataType*, const DataType*, const size_t&);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferArrayFromCpuToDevice(DataType*, const DataType*, const size_t&);
/* *************************************************************** */
extern "C++"
template <class DataType>
int cudaCommon_transferArrayFromDeviceToCpu(DataType*, const DataType*, const size_t&);
/* *************************************************************** */
using UniqueTextureObjectPtr = unique_ptr<cudaTextureObject_t, void(*)(cudaTextureObject_t*)>;
/* *************************************************************** */
extern "C++"
UniqueTextureObjectPtr cudaCommon_createTextureObject(const void *devPtr,
													  const cudaResourceType& resType,
													  const size_t& size = 0,
													  const cudaChannelFormatKind& channelFormat = cudaChannelFormatKindNone,
													  const unsigned& channelCount = 1,
													  const cudaTextureFilterMode& filterMode = cudaFilterModePoint,
													  const bool& normalizedCoordinates = false);
/* *************************************************************** */
