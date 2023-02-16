/** @file _reg_common_cuda.h
 * @author Marc Modat
 * @date 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "_reg_tools.h"

/* *************************************************************** */
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
struct __attribute__((aligned(4))) float4 {
    float x, y, z, w;
};
#endif
/* *************************************************************** */
#if CUDART_VERSION >= 3200
#   define NR_CUDA_SAFE_CALL(call) { \
		call; \
		cudaError err = cudaPeekAtLastError(); \
		if( cudaSuccess != err) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			reg_exit(); \
		} \
	}
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
		cudaDeviceSynchronize(); \
		cudaError err = cudaPeekAtLastError(); \
		if( err != cudaSuccess) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
			grid.x,grid.y,grid.z,block.x,block.y,block.z); \
			reg_exit(); \
		} \
		else{\
			printf("[NiftyReg CUDA DEBUG] kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n", \
			  cudaGetErrorString(cudaGetLastError()), grid.x, grid.y, grid.z, block.x, block.y, block.z);\
		}\
	}
#else //CUDART_VERSION >= 3200
#   define NR_CUDA_SAFE_CALL(call) { \
		call; \
		cudaError err = cudaDeviceSynchronize(); \
		if( cudaSuccess != err) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			reg_exit(); \
		} \
	}
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
		cudaError err = cudaDeviceSynchronize(); \
		if( err != cudaSuccess) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
			grid.x,grid.y,grid.z,block.x,block.y,block.z); \
			reg_exit(); \
		} \
	}
#endif //CUDART_VERSION >= 3200
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray**, int*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray**, cudaArray**, int*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE**, int);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE**, int*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE**, DTYPE**, int*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray*, cudaArray*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE*, DTYPE*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, DTYPE*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image*, DTYPE*, DTYPE*);
/* *************************************************************** */
extern "C++"
void cudaCommon_free(cudaArray*);
/* *************************************************************** */
extern "C++" template <class DTYPE>
void cudaCommon_free(DTYPE*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple(DTYPE*, nifti_image*);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple1(DTYPE*, DTYPE*, const unsigned);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToCpu(DTYPE*, DTYPE*, const unsigned int);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferArrayFromCpuToDevice(DTYPE*, DTYPE*, const unsigned int);
/* *************************************************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferArrayFromDeviceToCpu(DTYPE*, DTYPE*, const unsigned int);
/* *************************************************************** */
extern "C++"
void cudaCommon_destroyTextureObject(cudaTextureObject_t *texObj);
/* *************************************************************** */
using UniqueTextureObjectPtr = std::unique_ptr<cudaTextureObject_t, decltype(&cudaCommon_destroyTextureObject)>;
/* *************************************************************** */
extern "C++"
UniqueTextureObjectPtr cudaCommon_createTextureObject(void *devPtr,
													  cudaResourceType resType,
													  bool normalizedCoordinates = false,
													  size_t size = 0,
													  cudaChannelFormatKind channelFormat = cudaChannelFormatKindNone,
													  unsigned channelCount = 1,
													  cudaTextureFilterMode filterMode = cudaFilterModeLinear);
/* *************************************************************** */
