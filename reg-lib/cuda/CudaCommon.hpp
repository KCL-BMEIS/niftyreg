/** @file CudaCommon.hpp
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
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "_reg_tools.h"
#include "CudaContext.hpp"
#include "FloatOps.hpp"

/* *************************************************************** */
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
struct __attribute__((aligned(4))) float4 {
    float x, y, z, w;
};
#endif
/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
namespace Internal {
/* *************************************************************** */
inline void SafeCall(const std::string& file, const int line, const std::string& funcName) {
#if CUDART_VERSION >= 3200
	const cudaError_t err = cudaPeekAtLastError();
#else
	const cudaError_t err = cudaDeviceSynchronize();
#endif
	if (err != cudaSuccess)
        NiftyReg::Internal::FatalError(file, line, funcName, "CUDA error: "s + cudaGetErrorString(err));
}
/* *************************************************************** */
inline void CheckKernel(const std::string& file, const int line, const std::string& funcName, const dim3& grid, const dim3& block) {
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
template<class DataType>
void Allocate(DataType**, const size_t);
/* *************************************************************** */
template<class DataType>
void Allocate(DataType**, const int*);
/* *************************************************************** */
template<class DataType>
void Allocate(DataType**, DataType**, const int*);
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType*, const nifti_image*);
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType*, DataType*, const nifti_image*);
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType*, const DataType*, const size_t);
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToNifti(nifti_image*, const DataType*);
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToNifti(nifti_image*, const DataType*, const DataType*);
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToHost(DataType*, const DataType*, const size_t);
/* *************************************************************** */
template<class DataType>
void TransferFromHostToDevice(DataType*, const DataType*, const size_t);
/* *************************************************************** */
template<class DataType>
void Free(DataType*);
/* *************************************************************** */
namespace Internal {
template <class T>
struct UniquePtrDeleter { void operator()(T *ptr) const { Free(ptr); } };
}
/* *************************************************************** */
template<class T>
using UniquePtr = unique_ptr<T, Internal::UniquePtrDeleter<T>>;
/* *************************************************************** */
using UniqueTextureObjectPtr = UniquePtr<cudaTextureObject_t>;
/* *************************************************************** */
template<class DataType>
UniqueTextureObjectPtr CreateTextureObject(const DataType *devPtr,
                                           const size_t count,
                                           const cudaChannelFormatKind channelFormat,
                                           const unsigned channelCount);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
