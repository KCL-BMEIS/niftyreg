/*
 *  _reg_blockMatching_kernels.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_KERNELS_CU__
#define __REG_BLOCKMATCHING_KERNELS_CU__

//#define REDUCE reduceCustom1
#define REDUCE blockReduceSum

#include "assert.h"

// Some parameters that we need for the kernel execution.
// The caller is supposed to ensure that the values are set

// Number of blocks in each dimension
__device__ __constant__ int3 c_BlockDim;
__device__ __constant__ int c_StepSize;
__device__ __constant__ uint3 c_ImageSize;
__device__ __constant__ float r1c1;

// Transformation matrix from nifti header
__device__ __constant__ float4 t_m_a;
__device__ __constant__ float4 t_m_b;
__device__ __constant__ float4 t_m_c;

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define OVERLAP_SIZE 3
#define STEP_SIZE 1

#include "_reg_blockMatching_gpu.h"

texture<float, 1, cudaReadModeElementType> targetImageArray_texture;
texture<float, 1, cudaReadModeElementType> resultImageArray_texture;
texture<int, 1, cudaReadModeElementType> activeBlock_texture;

__device__ __inline__ float reduceCustom1(float data, const unsigned int tid, bool condition = false){
	static __shared__ float sData2[64];

	sData2[tid] = data;
	if (condition)printf("tid: %d | val: %f\n", tid, data);
	__syncthreads();
	/*const unsigned int laneId = tid % 32;
	const unsigned int warpid = tid / 32;*/


	/*assert(warpid == 0);
	assert(laneId == tid);
	assert(threadIdx.z<2);*/
	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

	/*else{
	assert(warpid == 1);
	assert(laneId != tid);
	assert(threadIdx.z>=2);
	}*/
	__syncthreads();
	return sData2[0];
}


/* *************************************************************** */
/* *************************************************************** */
__device__ __inline__ uint3 operator+(uint3 a, uint3 b) {
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ uint3 operator*(uint3 a, uint3 b){
	return make_uint3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float2 operator*(float a, float2 b){
	return make_float2(a*b.x, a*b.y);
}
__device__ float3 operator*(float a, float3 b){
	return make_float3(a*b.x, a*b.y, a*b.z);
}
__device__ float3 operator*(float3 a, float3 b){
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float4 operator*(float4 a, float4 b){
	return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
__device__ float4 operator*(float a, float4 b){
	return make_float4(a*b.x, a*b.y, a*b.z, 0.0f);
}
/* *************************************************************** */
__device__ float2 operator/(float2 a, float2 b){
	return make_float2(a.x / b.x, a.y / b.y);
}
__device__ float3 operator/(float3 a, float b){
	return make_float3(a.x / b, a.y / b, a.z / b);
}
__device__ float3 operator/(float3 a, float3 b){
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
/* *************************************************************** */
__device__ float2 operator+(float2 a, float2 b){
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float4 operator+(float4 a, float4 b){
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}
__device__ float3 operator+(float3 a, float3 b){
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
/* *************************************************************** */
__device__ float3 operator-(float3 a, float3 b){
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float4 operator-(float4 a, float4 b){
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.f);
}
/* *************************************************************** */
/* *************************************************************** */



// Apply the transformation matrix
__device__ inline void apply_affine(const float4 &pt, float * result)
{
	float4 mat = t_m_a;
	result[0] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
	mat = t_m_b;
	result[1] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
	mat = t_m_c;
	result[2] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
}
template <class DTYPE>
__device__ __inline__
void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
	out[0] = (DTYPE)mat[0 * 4 + 0] * in[0] +
		(DTYPE)mat[0 * 4 + 1] * in[1] +
		(DTYPE)mat[0 * 4 + 2] * in[2] +
		(DTYPE)mat[0 * 4 + 3];
	out[1] = (DTYPE)mat[1 * 4 + 0] * in[0] +
		(DTYPE)mat[1 * 4 + 1] * in[1] +
		(DTYPE)mat[1 * 4 + 2] * in[2] +
		(DTYPE)mat[1 * 4 + 3];
	out[2] = (DTYPE)mat[2 * 4 + 0] * in[0] +
		(DTYPE)mat[2 * 4 + 1] * in[1] +
		(DTYPE)mat[2 * 4 + 2] * in[2] +
		(DTYPE)mat[2 * 4 + 3];
	return;
}
//iterates through block numbers
__global__ void targetPosKernel(float *targetPosition_d, float* targetMatrix_xyz, int* mask, uint3 blockDims){

	const unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x);
	const unsigned int j = (blockIdx.y*blockDim.y + threadIdx.y);
	const unsigned int k = (blockIdx.z*blockDim.z + threadIdx.z);



	//if ((i < 23) && (j < 28) && (k < 23)){
	assert(k < blockDims.x);
	assert(j < blockDims.y);
	assert(i < blockDims.x);
	const unsigned int flatIdx = k*blockDims.x * blockDims.y + j*blockDims.x + i;

	float targetPosition_temp[3] = { i* BLOCK_WIDTH, j* BLOCK_WIDTH, k* BLOCK_WIDTH };
	float tempPosition[3];

	//bool is800 = (i == 8 && j == 0 && k == 0);
	reg_mat44_mul_cuda<float>(targetMatrix_xyz, targetPosition_temp, tempPosition);
	/*if (is800) printf("cuda (8,0,0): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
	if (is800) printf("cuda (8,0,0): %d-%d\n", flatIdx, activeBlock[flatIdx]);*/



	//const unsigned int z = 3 * params->definedActiveBlock;
	//const unsigned int targetIndex = 3 * /*(k*BLOCK_WIDTH*BLOCK_WIDTH + j*BLOCK_WIDTH + i)*/ flatIdx;
	const int activeBlock = tex1Dfetch(activeBlock_texture, flatIdx);
	//assert(activeBlock < blockDims.x * blockDims.y* blockDims.z / 2);

	if (activeBlock != -1){
		const unsigned int active = 3 * /*activeBlock[flatIdx]*/activeBlock;



		targetPosition_d[active] = tempPosition[0];
		targetPosition_d[active + 1] = tempPosition[1];
		targetPosition_d[active + 2] = tempPosition[2];
		/*if (activeBlock[flatIdx] == 0 && !is800){
			printf("cuda (x,x,x): %d-%d-%d: %d-%d\n", i, j, k, flatIdx, activeBlock[flatIdx]);
			}*/
	}
	//}
}


// CUDA kernel to process the target values
__global__ void targetBlocksKernel(float *targetPosition_d, float *targetValues)
{

	const unsigned int blockId = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;
	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, blockId);

	const unsigned int targetIndex_start_x = blockIdx.x * blockDim.x;
	const unsigned int targetIndex_start_y = blockIdx.y * blockDim.y;
	const unsigned int targetIndex_start_z = blockIdx.z * blockDim.z;

	const unsigned int x = targetIndex_start_x + threadIdx.x;
	const unsigned int y = targetIndex_start_y + threadIdx.y;
	const unsigned int z = targetIndex_start_z + threadIdx.z;

	const uint3 imageSize = c_ImageSize;

	const bool valid = (currentBlockIndex >= 0 && z >= 0 && z < imageSize.z) && (y >= 0 && y < imageSize.y) && (x >= 0 && x < imageSize.x);
	if (valid){
		// Get the corresponding (i, j, k) indices

		const unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
		const unsigned int offset = blockId * blockSize;//currentBlockIndex?

		/*int targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;
		int targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;
		int targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;*/

		const unsigned int i = blockIdx.x;
		const unsigned int j = blockIdx.y;
		const unsigned int k = blockIdx.z;

		const unsigned int index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x  * blockDim.y;//0-blockSize
		const unsigned int indexXYZ = x + y           * imageSize.x + z * imageSize.x * imageSize.y;//base_linear_index - base_linear_index + blockSize
		/*if (blockIdx.x == 5 && blockIdx.y == 0 && blockIdx.z == 4)
			printf("idx: %d | bidx: %d | tstIdx: %d\n", index, blockId, indexXYZ);*/

		targetValues[index + offset] = tex1Dfetch(targetImageArray_texture, indexXYZ);
		assert(index + offset == indexXYZ);

		bool is000 = (i == 4) && (j == 5) && (k == 6);
		//if (is000) if (is000) printf("CUDA: targetIndex: %d | bIdx: %d | ijk: %d-%d-%d | xyz: %d-%d-%d %f\n", index /*+ offset*/, blockId, i, j, k, x, y, z, targetValues[index + offset]);


		/*float4 targetPosition;
		targetPosition.x = i * BLOCK_WIDTH;
		targetPosition.y = j * BLOCK_WIDTH;
		targetPosition.z = k * BLOCK_WIDTH;
		apply_affine(targetPosition, &(targetPosition_d[blockId * 3]));*/
	}

}

//original code!!!!!!!!!!!
// CUDA kernel to process the target values
__global__ void process_target_blocks_gpu(float *targetPosition_d,
	float *targetValues)
{
	const int tid = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
	const int3 bDim = c_BlockDim;

	if (tid < bDim.x * bDim.y * bDim.z){
		const int currentBlockIndex = tex1Dfetch(activeBlock_texture, tid);
		if (currentBlockIndex >= 0){
			// Get the corresponding (i, j, k) indices
			int tempIndex = currentBlockIndex;
			const int k = (int)(tempIndex / (bDim.x * bDim.y));
			tempIndex -= k * bDim.x * bDim.y;
			const int j = (int)(tempIndex / (bDim.x));
			const int i = tempIndex - j * (bDim.x);
			const int offset = tid * BLOCK_SIZE;
			const int targetIndex_start_x = i * BLOCK_WIDTH;
			const int targetIndex_start_y = j * BLOCK_WIDTH;
			const int targetIndex_start_z = k * BLOCK_WIDTH;

			int targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;
			int targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;
			int targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;
			const uint3 imageSize = c_ImageSize;
			for (int count = 0; count < BLOCK_SIZE; ++count) targetValues[count + offset] = 0.0f;
			unsigned int index = 0;

			for (int z = targetIndex_start_z; z < targetIndex_end_z; ++z){
				if (z >= 0 && z < imageSize.z) {
					int indexZ = z * imageSize.x * imageSize.y;
					for (int y = targetIndex_start_y; y < targetIndex_end_y; ++y){
						if (y >= 0 && y < imageSize.y) {
							int indexXYZ = indexZ + y * imageSize.x + targetIndex_start_x;
							for (int x = targetIndex_start_x; x < targetIndex_end_x; ++x){
								if (x >= 0 && x < imageSize.x) {
									targetValues[index + offset] = tex1Dfetch(targetImageArray_texture, indexXYZ);
								}
								indexXYZ++;
								index++;
							}
						}
						else index += BLOCK_WIDTH;
					}
				}
				else index += BLOCK_WIDTH * BLOCK_WIDTH;
			}

			float4 targetPosition;
			targetPosition.x = i * BLOCK_WIDTH;
			targetPosition.y = j * BLOCK_WIDTH;
			targetPosition.z = k * BLOCK_WIDTH;
			apply_affine(targetPosition, &(targetPosition_d[tid * 3]));
		}
	}
}

//original
// CUDA kernel to process the result blocks
__global__ void resultBlocksKernel(float *resultPosition_d, float *targetValues){




	const int tid = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
	const int3 bDim = c_BlockDim;
	int tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
	__shared__ int ctid;
	if (tempIndex == 0) ctid = (int)(tid / NUM_BLOCKS_TO_COMPARE);
	__syncthreads();
	//const int ctid = (int)(tid / NUM_BLOCKS_TO_COMPARE);
	__shared__ float4 localCC[NUM_BLOCKS_TO_COMPARE];
	__shared__ int3 indexes;
	localCC[tempIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	__shared__ int updateThreadID;
	updateThreadID = -1;
	if (ctid < bDim.x * bDim.y * bDim.z) {
		const int activeBlockIndex = tex1Dfetch(activeBlock_texture, ctid);
		tempIndex = activeBlockIndex;
		int k = (int)(tempIndex / (bDim.x * bDim.y));
		tempIndex -= k * bDim.x * bDim.y;
		int j = (int)(tempIndex / (bDim.x));
		int i = tempIndex - j * (bDim.x);
		tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
		if (tempIndex == 0) {
			indexes.x = i * BLOCK_WIDTH;
			indexes.y = j * BLOCK_WIDTH;
			indexes.z = k * BLOCK_WIDTH;
		}
		__syncthreads();

		if (activeBlockIndex >= 0) {
			const int block_offset = ctid * BLOCK_SIZE;
			const uint3 imageSize = c_ImageSize;
			int k = (int)tempIndex / NUM_BLOCKS_TO_COMPARE_2D;
			tempIndex -= k * NUM_BLOCKS_TO_COMPARE_2D;
			int j = (int)tempIndex / NUM_BLOCKS_TO_COMPARE_1D;
			int i = tempIndex - j * NUM_BLOCKS_TO_COMPARE_1D;
			k -= OVERLAP_SIZE;
			j -= OVERLAP_SIZE;
			i -= OVERLAP_SIZE;
			tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
			int resultIndex_start_z = indexes.z + k;
			int resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;
			int resultIndex_start_y = indexes.y + j;
			int resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
			int resultIndex_start_x = indexes.x + i;
			int resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
			__shared__ float4 cc_vars[NUM_BLOCKS_TO_COMPARE];
			cc_vars[tempIndex].x = 0.0f;
			cc_vars[tempIndex].y = 0.0f;
			unsigned int index = 0;
			for (int z = resultIndex_start_z; z < resultIndex_end_z; ++z){
				if (z >= 0 && z < imageSize.z) {
					int indexZ = z * imageSize.y * imageSize.x;
					for (int y = resultIndex_start_y; y < resultIndex_end_y; ++y){
						if (y >= 0 && y < imageSize.y) {
							int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
							for (int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
								if (x >= 0 && x < imageSize.x) {
									cc_vars[tempIndex].x = tex1Dfetch(resultImageArray_texture, indexXYZ);
									cc_vars[tempIndex].y = targetValues[block_offset + index];
									if (cc_vars[tempIndex].x != 0.0f && cc_vars[tempIndex].y != 0.0f) {
										localCC[tempIndex].x += cc_vars[tempIndex].x;
										localCC[tempIndex].y += cc_vars[tempIndex].y;
										++localCC[tempIndex].z;
									}
								}
								++indexXYZ;
								++index;
							}
						}
						else index += BLOCK_WIDTH;
					}
				}
				else index += BLOCK_WIDTH * BLOCK_WIDTH;
			}

			if (localCC[tempIndex].z > 0) {
				localCC[tempIndex].x /= localCC[tempIndex].z;
				localCC[tempIndex].y /= localCC[tempIndex].z;
			}
			cc_vars[tempIndex].z = 0.0f;
			cc_vars[tempIndex].w = 0.0f;
			index = 0;
			for (int z = resultIndex_start_z; z < resultIndex_end_z; ++z){
				if (z >= 0 && z < imageSize.z) {
					int indexZ = z * imageSize.y * imageSize.x;
					for (int y = resultIndex_start_y; y < resultIndex_end_y; ++y){
						if (y >= 0 && y < imageSize.y) {
							int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
							for (int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
								if (x >= 0 && x < imageSize.x) {
									cc_vars[tempIndex].x = tex1Dfetch(resultImageArray_texture, indexXYZ);
									cc_vars[tempIndex].y = targetValues[block_offset + index];
									if (cc_vars[tempIndex].x != 0.0f && cc_vars[tempIndex].y != 0.0f) {
										cc_vars[tempIndex].x -= localCC[tempIndex].x;
										cc_vars[tempIndex].y -= localCC[tempIndex].y;

										cc_vars[tempIndex].z += cc_vars[tempIndex].x * cc_vars[tempIndex].x;
										cc_vars[tempIndex].w += cc_vars[tempIndex].y * cc_vars[tempIndex].y;
										localCC[tempIndex].w += cc_vars[tempIndex].x * cc_vars[tempIndex].y;
									}
								}
								++indexXYZ;
								++index;
							}
						}
						else index += BLOCK_WIDTH;
					}
				}
				else index += BLOCK_WIDTH * BLOCK_WIDTH;
			}

			if (localCC[tempIndex].z > (float)(BLOCK_SIZE / 2)) {
				if (cc_vars[tempIndex].z > 0.0f && cc_vars[tempIndex].w > 0.0f) {
					localCC[tempIndex].w = fabsf(localCC[tempIndex].w / sqrt(cc_vars[tempIndex].z * cc_vars[tempIndex].w));
				}
			}
			else { localCC[tempIndex].w = 0.0f; }

			localCC[tempIndex].x = i;
			localCC[tempIndex].y = j;
			localCC[tempIndex].z = k;

			// Just take ownership of updating the final value
			if (updateThreadID == -1) updateThreadID = tid;
		}
		__syncthreads();

		// Just let one thread do the final update
		if (tid == updateThreadID) {
			__shared__ float4 bestCC;
			bestCC = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for (int i = 0; i < NUM_BLOCKS_TO_COMPARE; ++i) {
				if (localCC[i].w > bestCC.w) {
					bestCC.x = localCC[i].x;
					bestCC.y = localCC[i].y;
					bestCC.z = localCC[i].z;
					bestCC.w = localCC[i].w;
				}
			}
			bestCC.x += indexes.x;
			bestCC.y += indexes.y;
			bestCC.z += indexes.z;
			apply_affine(bestCC, &(resultPosition_d[ctid * 3]));
		}
	}
}

//__global__ void resultsKernel(float *resultPosition, float *targetValues, int* mask, float* targetMatrix_xyz, int3 blockDims, const float overlapSize){
//
//	__shared__ float sResultValues[4096];
//	__shared__ float sTargetValues[64];
//
//	__shared__ float targetMean = 0.0;
//	__shared__ float resultMean = 0.0;
//
//
//	const unsigned int blockSize = blockDims.x * blockDims.y * blockDims.z;
//	const unsigned int blockId = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;
//	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, blockId);
//
//	const unsigned int flatIdx = threadIdx.z*blockDim.x * blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
//
//
//
//
//	if (currentBlockIndex >= 0){
//		const uint3 imageSize = c_ImageSize;
//
//		const unsigned int i = blockIdx.x;
//		const unsigned int j = blockIdx.y;
//		const unsigned int k = blockIdx.z;
//
//		const unsigned int xDelta = overlapSize / 2;
//		const unsigned int yDelta = overlapSize / 2;
//		const unsigned int zDelta = overlapSize / 2;
//
//		const  int l = threadIdx.x - xDelta;
//		const  int m = threadIdx.y - yDelta;
//		const  int n = threadIdx.z - zDelta;
//
//		const unsigned int xBase = blockIdx.x * blockDims.x;
//		const unsigned int yBase = blockIdx.y * blockDims.y;
//		const unsigned int zBase = blockIdx.z * blockDims.z;
//
//		const  int xOverlap = xBase + l;
//		const  int yOverlap = yBase + m;
//		const  int zOverlap = zBase + n;
//
//		const  int xImage = xBase + ceilf(threadIdx.x / overlapSize);
//		const  int yImage = yBase + ceilf(threadIdx.x / overlapSize);
//		const  int zImage = zBase + ceilf(threadIdx.x / overlapSize);
//
//		//assert(currentBlockIndex < gridDim.x* gridDim.y*gridDim.z / 2);
//
//		//shared mem block values?
//		float bestDisplacement[3], targetPosition_temp[3], tempPosition[3];
//		float bestCC = 0.0;
//		bestDisplacement[0] = /*nanf("NaNs")*/0.0f;//possible error?
//		bestDisplacement[1] = 0.f;
//		bestDisplacement[2] = 0.f;
//
//		/*assert(blockId < gridDim.x* gridDim.y*gridDim.z);
//		assert(currentBlockIndex < blockDims.x * blockDims.y* blockDims.z / 2);
//		assert(blockDims.x == gridDim.x);
//		assert(blockDims.y == gridDim.y);
//		assert(blockDims.y == gridDim.y);
//
//		assert(xBase >= 0);
//		assert(yBase >= 0);
//		assert(zBase >= 0);*/
//
//
//
//		const bool threadInImage = ((xOverlap < imageSize.x) && (yOverlap < imageSize.y) && (zOverlap < imageSize.z)) && ((xOverlap >= 0) && (yOverlap >= 0) && (zOverlap >= 0));
//		const unsigned int resultIndex = xImage + yImage * blockDim.x + zImage * blockDim.x  * blockDim.y;//?
//		const unsigned int offset = blockId * blockSize;//currentBlockIndex?
//		if (threadInImage){
//
//
//			/*assert(offset + resultIndex < imageSize.x*imageSize.y*imageSize.z);
//			assert(resultIndex < blockDim.x*blockDim.y*blockDim.z);*/
//
//			sTargetValues[resultIndex] = targetValues[offset + resultIndex];
//			/*assert(x >= 0);
//			assert(y >= 0);
//			assert(z >= 0);
//
//
//			assert(x < imageSize.x);
//			assert(y < imageSize.y);
//			assert(z < imageSize.z);*/
//			const unsigned int indexXYZ = xImage + yImage           * imageSize.x + zImage * imageSize.x * imageSize.y;
//			//assert(indexXYZ < imageSize.x*imageSize.y*imageSize.z);
//			//those on shared mem
//			const float value = tex1Dfetch(resultImageArray_texture, indexXYZ);//change
//			int maskVal = mask[indexXYZ];
//			if (maskVal >= 0 && value == value){ sResultValues[resultIndex] = value; }
//		}
//		__syncthreads();
//
//
//
//
//
//		//reduce over sMem here
//		reduce(sTargetValues, flatIdx, BLOCK_SIZE);
//		reduce(sResultValues, flatIdx, BLOCK_SIZE);
//
//
//		if (flatIdx == 0) targetMean = sTargetValues[0];
//		if (flatIdx == 0) resultMean = sTargetValues[0];
//		__syncthreads();
//
//		targetMean /= BLOCK_SIZE;
//		resultMean /= BLOCK_SIZE;
//
//		__shared__ float targetVar = 0.0;
//		__shared__ float resultVar = 0.0;
//		__shared__ float localCC = 0.0;
//
//		sTargetValues[flatIdx] -= targetMean;
//		sResultValues[flatIdx] -= resultMean;
//
//		if (flatIdx == 0) targetVar = sTargetValues[0] * sTargetValues[0];
//		if (flatIdx == 0) resultVar = sResultValues[0] * sResultValues[0];
//		if (flatIdx == 0) localCC = sTargetValues[0] * sResultValues[0];
//
//
//
//		if (flatIdx == 0) localCC = fabs(localCC / sqrtf(targetVar*resultVar));
//
//
//		//warp vote here
//		if (localCC > bestCC) {
//			bestCC = localCC;
//			bestDisplacement[0] = (float)l;
//			bestDisplacement[1] = (float)m;
//			bestDisplacement[2] = (float)n;
//		}
//		/*
//		}
//		}*/
//
//		if (bestDisplacement[0] == bestDisplacement[0]) {
//			/*assert(params->activeBlock[blockIndex] > -1);
//			printf("%d-%d\n", params->activeBlock[blockIndex] , params->definedActiveBlock);
//			assert(params->activeBlock[blockIndex] == params->definedActiveBlock + 1);*/
//			targetPosition_temp[0] = (float)(i*BLOCK_WIDTH);
//			targetPosition_temp[1] = (float)(j*BLOCK_WIDTH);
//			targetPosition_temp[2] = (float)(k*BLOCK_WIDTH);
//
//			bestDisplacement[0] += targetPosition_temp[0];
//			bestDisplacement[1] += targetPosition_temp[1];
//			bestDisplacement[2] += targetPosition_temp[2];
//
//
//			/*if (is800) printf("cpu (8,0, 0): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
//			if (is800) printf("cpu (8,0,0): %d-%d\n", blockIndex, params->activeBlock[blockIndex]);
//			if (params->activeBlock[blockIndex] == 0)
//			printf("cpu (x,x,x): %d-%d-%d  %d\n", i, j, k, blockIndex);*/
//
//			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);
//
//			const unsigned int posIdx = 3 * currentBlockIndex;
//
//			//assert(currentBlockIndex >= 0);
//			resultPosition[posIdx] = tempPosition[0];
//			resultPosition[posIdx + 1] = tempPosition[1];
//			resultPosition[posIdx + 2] = tempPosition[2];
//
//			//std::cout << "ijk: " << i << "-" << j << "-" << k << " idx: " << params->definedActiveBlock << std::endl;
//
//
//		}
//	}
//
//
//
//}
__device__ __inline__ void reduceCC(float* sData, const unsigned int tid, const unsigned int blockSize){


	if (blockSize >= 512) { if (tid < 256) { sData[tid] += sData[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sData[tid] += sData[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sData[tid] += sData[tid + 64]; } __syncthreads(); }
	if (tid < 32){
		if (blockSize >= 64) sData[tid] += sData[tid + 32];
		if (blockSize >= 32) sData[tid] += sData[tid + 16];
		if (blockSize >= 16) sData[tid] += sData[tid + 8];
		if (blockSize >= 8) sData[tid] += sData[tid + 4];
		if (blockSize >= 4) sData[tid] += sData[tid + 2];
		if (blockSize >= 2) sData[tid] += sData[tid + 1];
	}
}

__device__ __inline__ void reduce(float* sData, const unsigned int tid, const unsigned int blockSize){


	if (blockSize >= 512) { if (tid < 256) { sData[tid] += sData[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sData[tid] += sData[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sData[tid] += sData[tid + 64]; } __syncthreads(); }
	if (tid < 32){
		if (blockSize >= 64) sData[tid] += sData[tid + 32];
		if (blockSize >= 32) sData[tid] += sData[tid + 16];
		if (blockSize >= 16) sData[tid] += sData[tid + 8];
		if (blockSize >= 8) sData[tid] += sData[tid + 4];
		if (blockSize >= 4) sData[tid] += sData[tid + 2];
		if (blockSize >= 2) sData[tid] += sData[tid + 1];
	}
}

//must parameterize warpsize in both cuda and cl
__device__ __inline__ float reduceCustom_f1(float data, const unsigned int tid, const unsigned int blockSize){
	static __shared__ float sDataBuff[8 * 8 * 8];

	sDataBuff[tid] = data;
	__syncthreads();


	const unsigned int laneId = tid % 32;
	const unsigned int warpId = tid / 32;
	const unsigned int bid = tid / blockSize;

	if (warpId % 2 == 0){
		sDataBuff[tid] += sDataBuff[tid + 32];
		sDataBuff[tid] += sDataBuff[tid + 16];
		sDataBuff[tid] += sDataBuff[tid + 8];
		sDataBuff[tid] += sDataBuff[tid + 4];
		sDataBuff[tid] += sDataBuff[tid + 2];
		sDataBuff[tid] += sDataBuff[tid + 1];
	}



	__syncthreads();
	return sDataBuff[bid * blockSize];
}

__device__ __inline__ float reduceCustom_f(float data, const unsigned int tid){
	static __shared__ float sData2[64];

	sData2[tid] = data;
	__syncthreads();

	if (tid < 32){
		sData2[tid] += sData2[tid + 32];
		sData2[tid] += sData2[tid + 16];
		sData2[tid] += sData2[tid + 8];
		sData2[tid] += sData2[tid + 4];
		sData2[tid] += sData2[tid + 2];
		sData2[tid] += sData2[tid + 1];
	}



	__syncthreads();
	return sData2[0];
}

__device__ __inline__ float reduceCustom(float data, const unsigned int tid){
	static __shared__ float sData2[64];

	sData2[tid] = data;
	__syncthreads();
	/*const unsigned int laneId = tid % 32;
	const unsigned int warpid = tid / 32;*/


	/*assert(warpid == 0);
	assert(laneId == tid);
	assert(threadIdx.z<2);*/
	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

	/*else{
		assert(warpid == 1);
		assert(laneId != tid);
		assert(threadIdx.z>=2);
		}*/
	__syncthreads();
	return sData2[0];
}



__inline__ __device__
float warpAllReduceSum(float val) {
	for (int mask = 16; mask > 0; mask /= 2)
		val += __shfl_xor(val, mask);
	return val;
}

__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = 16; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}
__device__ __inline__ float reduceCustom2(float data, const unsigned int tid){
	static __shared__ float sData1[32];

	if (tid >= 32)
		sData1[tid - 32] = data;
	__syncthreads();




	if (tid < 32) sData1[0] = warpAllReduceSum(data + sData1[tid]);
	__syncthreads();
	return sData1[0];
}

__inline__ __device__
float blockReduceSum(float val, int tid) {



	static __shared__ float shared[2]; // Shared mem for 32 partial sums
	int lane = tid % 32;
	int wid = tid / 32;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val;	// Write reduced value to shared memory
	//if (blockIdx.x == 8 && blockIdx.y == 0 && blockIdx.z == 0) printf("idx: %d | lane: %d \n", tid, lane);
	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = shared[0] + shared[1];

	return val;
}


//launched as 4x4x4 blocks
//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernel(float *resultPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){

	__shared__ float sResultValues[12 * 12 * 12];

	const unsigned int i = blockIdx.x;
	const unsigned int j = blockIdx.y;
	const unsigned int k = blockIdx.z;
	//bool is800 = (i == 8 && j == 0 && k == 0);
	/*assert(i < blockDims.x);
	assert(j < blockDims.y);
	assert(k < blockDims.z);*/


	const unsigned int bid = i + gridDim.x * j + (gridDim.x * gridDim.y) * k;
	//assert(blockIdFlat < (blockDims.x)*(blockDims.y)*(blockDims.z));


	const unsigned int xBaseImage = i * blockDim.x;
	const unsigned int yBaseImage = j * blockDim.y;
	const unsigned int zBaseImage = k * blockDim.z;

	/*assert(xBaseImage < imageSize.x);
	assert(yBaseImage < imageSize.y);
	assert(zBaseImage < imageSize.z);*/

	const unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x  * blockDim.y;//0-blockSize
	//assert(kernelIdxFlat < 64);



	const unsigned int xImage = xBaseImage + threadIdx.x;
	const unsigned int yImage = yBaseImage + threadIdx.y;
	const unsigned int zImage = zBaseImage + threadIdx.z;



	const unsigned long idx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);


	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
	//assert(currentBlockIndex < (blockDims.x )*(blockDims.y)*(blockDims.z));
	if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
		//assert(indexXYZ < imageSize.x*imageSize.y*imageSize.z);
		//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);
		/*assert(xImage < imageSize.x);
		assert(yImage < imageSize.y);
		assert(zImage < imageSize.z);*/
		for (int n = -1; n <= 1; n += 1)
		{
			for (int m = -1; m <= 1; m += 1)
			{
				for (int l = -1; l <= 1; l += 1)
				{


					const int x = l * 4 + threadIdx.x;
					const int y = m * 4 + threadIdx.y;
					const int z = n * 4 + threadIdx.z;

					const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);
					/*assert(sIdx  >= 0);
					assert(sIdx  <12 * 12 * 12);*/

					/*assert(x >= -4);
					assert(y >= -4);
					assert(z >= -4);

					assert(x < 8);
					assert(y < 8);
					assert(z < 8);*/



					const unsigned int xImageIn = xBaseImage + x;
					const unsigned int yImageIn = yBaseImage + y;
					const unsigned int zImageIn = zBaseImage + z;


					/*assert(xImageIn < imageSize.x);
					assert(yImageIn < imageSize.y);
					assert(zImageIn < imageSize.z);

					assert(xImageIn >= 0);
					assert(yImageIn >= 0);
					assert(zImageIn >= 0);*/

					const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);
					/*assert(indexXYZIn < imageSize.x*imageSize.y*imageSize.z);
					assert(indexXYZIn >= 0);*/
					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
					if (valid)
						sResultValues[sIdx] = tex1Dfetch(resultImageArray_texture, indexXYZIn);
					//if (is800 && tid == 0 && l==0 && m==0 && n==0) printf("gpu 800 s1ResultValues: %d: %f | id: %d\n", tid, sResultValues[sIdx], sIdx);


				}
			}
		}
		float rTargetValue = tex1Dfetch(targetImageArray_texture, idx);
		//if (is800 && tid == 0) printf("gpu 800 rTargetValue: %d: %f | id: %d\n", tid, rTargetValue, idx);
		//reduce(sTargetValues, kernelIdxFlat, BLOCK_SIZE);
		const float rSumTargetValues = REDUCE(rTargetValue, tid);
		const float rTargetMean = rSumTargetValues / BLOCK_SIZE;
		const float rTargetTemp = (rTargetValue - rTargetMean);
		const float rTargetVar = REDUCE(rTargetTemp*rTargetTemp, tid);
		//if (is800 && tid == 0) printf("gpu 800 targetVar: %d: %f\n",tid,  rTargetVar);


		float bestDisplacement[3];
		float bestCC = 0.0;


		// iteration over the result blocks
		for (unsigned int n = 1; n < 8; n += 1)
		{
			for (unsigned int m = 1; m < 8; m += 1)
			{
				for (unsigned int l = 1; l < 8; l += 1)
				{

					const unsigned int x = threadIdx.x + l;
					const unsigned int y = threadIdx.y + m;
					const unsigned int z = threadIdx.z + n;


					const unsigned int sIdxIn = z * 12 * 12 + y * 12 + x;
					/*assert(sIdxIn  >= 0);
					assert(sIdxIn  <12 * 12 * 12);*/


					const float rResultValue = sResultValues[sIdxIn];

					const float resultMean = REDUCE(rResultValue, tid) / BLOCK_SIZE;
					const float resultTemp = rResultValue - resultMean;
					const float resultVar = REDUCE(resultTemp*resultTemp, tid);
					const float sumTargetResult = REDUCE((rTargetTemp)*(resultTemp), tid);

					const float localCC = fabs((sumTargetResult) / sqrtf(rTargetVar*resultVar));
					/*if (is800 && tid == 0 && l == 4 && m ==4 && n == 4) printf("gpu 800 bestCC: %d-%d-%d: %f\n", l, m, n, bestCC);
					if (is800 && tid == 0 && l == 4 && m == 4 && n == 4) printf("gpu 800 val: %d-%d-%d: %f\n", l, m, n, rResultValue);*/

					//warp vote here
					if (tid == 0 && localCC > bestCC) {
						bestCC = localCC;
						bestDisplacement[0] = l - 4;
						bestDisplacement[1] = m - 4;
						bestDisplacement[2] = n - 4;
					}

				}
			}
		}
		__syncthreads();
		if (tid == 0) {

			float  tempPosition[3];



			bestDisplacement[0] += (i*BLOCK_WIDTH);
			bestDisplacement[1] += (j*BLOCK_WIDTH);
			bestDisplacement[2] += (k*BLOCK_WIDTH);



			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);


			/*if (is800) printf("gpu (8,0, 0): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
			if (is800) printf("gpu (8,0,0): %d-%d\n", bid, currentBlockIndex);
			if (currentBlockIndex == 0)
			printf("gpu (x,x,x): %d-%d-%d  %d\n", i, j, k, bid);*/

			const unsigned int posIdx = 3 * currentBlockIndex;

			/*bool is66 = posIdx == 66/3;
			if (is66){
			printf("gpu 66 (x,x,x): %d-%d-%d  %d\n", i, j, k, bid);
			printf("gpu 66 (x,x, x): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
			}*/

			//assert(isfinite(tempPosition[0]));

			//assert(currentBlockIndex >= 0);
			resultPosition[posIdx] = tempPosition[0];
			resultPosition[posIdx + 1] = tempPosition[1];
			resultPosition[posIdx + 2] = tempPosition[2];

			//std::cout << "ijk: " << i << "-" << j << "-" << k << " idx: " << params->definedActiveBlock << std::endl;


		}

	}

}

//launched as 64 thread blocks
//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernel2(float *resultPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){

	__shared__ float sResultValues[12 * 12 * 12];

	const unsigned int i = blockIdx.x;
	const unsigned int j = blockIdx.y;
	const unsigned int k = blockIdx.z;

	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;
	/*assert(idx < 4);
	assert(idy < 4);
	assert(idz < 4);*/
	//bool is800 = (i == 8 && j == 0 && k == 0);
	/*assert(i < blockDims.x);
	assert(j < blockDims.y);
	assert(k < blockDims.z);*/


	const unsigned int bid = i + gridDim.x * j + (gridDim.x * gridDim.y) * k;
	//assert(blockIdFlat < (blockDims.x)*(blockDims.y)*(blockDims.z));


	const unsigned int xBaseImage = i * 4;
	const unsigned int yBaseImage = j * 4;
	const unsigned int zBaseImage = k * 4;

	/*assert(xBaseImage < imageSize.x);
	assert(yBaseImage < imageSize.y);
	assert(zBaseImage < imageSize.z);*/

	const unsigned int tid = threadIdx.x;//0-blockSize
	//assert(kernelIdxFlat < 64);



	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;



	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);


	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
	//assert(currentBlockIndex < (blockDims.x )*(blockDims.y)*(blockDims.z));
	if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
		//assert(indexXYZ < imageSize.x*imageSize.y*imageSize.z);
		//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);
		/*assert(xImage < imageSize.x);
		assert(yImage < imageSize.y);
		assert(zImage < imageSize.z);*/
		for (int n = -1; n <= 1; n += 1)
		{
			for (int m = -1; m <= 1; m += 1)
			{
				for (int l = -1; l <= 1; l += 1)
				{


					const int x = l * 4 + idx;
					const int y = m * 4 + idy;
					const int z = n * 4 + idz;

					const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);
					/*assert(sIdx  >= 0);
					assert(sIdx  <12 * 12 * 12);*/

					/*assert(x >= -4);
					assert(y >= -4);
					assert(z >= -4);

					assert(x < 8);
					assert(y < 8);
					assert(z < 8);*/



					const unsigned int xImageIn = xBaseImage + x;
					const unsigned int yImageIn = yBaseImage + y;
					const unsigned int zImageIn = zBaseImage + z;


					/*assert(xImageIn < imageSize.x);
					assert(yImageIn < imageSize.y);
					assert(zImageIn < imageSize.z);

					assert(xImageIn >= 0);
					assert(yImageIn >= 0);
					assert(zImageIn >= 0);*/

					const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);
					/*assert(indexXYZIn < imageSize.x*imageSize.y*imageSize.z);
					assert(indexXYZIn >= 0);*/
					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
					if (valid)
						sResultValues[sIdx] = tex1Dfetch(resultImageArray_texture, indexXYZIn);
					//if (is800 && tid == 0 && l==0 && m==0 && n==0) printf("gpu 800 s1ResultValues: %d: %f | id: %d\n", tid, sResultValues[sIdx], sIdx);


				}
			}
		}
		float rTargetValue = tex1Dfetch(targetImageArray_texture, imgIdx);
		//if (is800 && tid == 0) printf("gpu 800 rTargetValue: %d: %f | id: %d\n", tid, rTargetValue, idx);
		//reduce(sTargetValues, kernelIdxFlat, BLOCK_SIZE);
		const float rSumTargetValues = REDUCE(rTargetValue, tid);
		const float rTargetMean = rSumTargetValues / BLOCK_SIZE;
		const float rTargetTemp = (rTargetValue - rTargetMean);
		const float rTargetVar = REDUCE(rTargetTemp*rTargetTemp, tid);
		//if (is800 && tid == 0) printf("gpu 800 targetVar: %d: %f\n",tid,  rTargetVar);


		float bestDisplacement[3];
		float bestCC = 0.0;


		// iteration over the result blocks
		for (unsigned int n = 1; n < 8; n += 1)
		{
			for (unsigned int m = 1; m < 8; m += 1)
			{
				for (unsigned int l = 1; l < 8; l += 1)
				{

					const unsigned int x = idx + l;
					const unsigned int y = idy + m;
					const unsigned int z = idz + n;


					const unsigned int sIdxIn = z * 12 * 12 + y * 12 + x;
					/*assert(sIdxIn  >= 0);
					assert(sIdxIn  <12 * 12 * 12);*/


					const float rResultValue = sResultValues[sIdxIn];

					const float resultMean = REDUCE(rResultValue, tid) / BLOCK_SIZE;
					const float resultTemp = rResultValue - resultMean;
					const float resultVar = REDUCE(resultTemp*resultTemp, tid);
					const float sumTargetResult = REDUCE((rTargetTemp)*(resultTemp), tid);

					const float localCC = fabs((sumTargetResult) / sqrtf(rTargetVar*resultVar));
					/*if (is800 && tid == 0 && l == 4 && m ==4 && n == 4) printf("gpu 800 bestCC: %d-%d-%d: %f\n", l, m, n, bestCC);
					if (is800 && tid == 0 && l == 4 && m == 4 && n == 4) printf("gpu 800 val: %d-%d-%d: %f\n", l, m, n, rResultValue);*/

					//warp vote here
					if (tid == 0 && localCC > bestCC) {
						bestCC = localCC;
						bestDisplacement[0] = l - 4;
						bestDisplacement[1] = m - 4;
						bestDisplacement[2] = n - 4;
					}

				}
			}
		}
		__syncthreads();
		if (tid == 0) {

			float  tempPosition[3];



			bestDisplacement[0] += (i*BLOCK_WIDTH);
			bestDisplacement[1] += (j*BLOCK_WIDTH);
			bestDisplacement[2] += (k*BLOCK_WIDTH);



			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);


			/*if (is800) printf("gpu (8,0, 0): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
			if (is800) printf("gpu (8,0,0): %d-%d\n", bid, currentBlockIndex);
			if (currentBlockIndex == 0)
			printf("gpu (x,x,x): %d-%d-%d  %d\n", i, j, k, bid);*/

			const unsigned int posIdx = 3 * currentBlockIndex;

			/*bool is66 = posIdx == 66/3;
			if (is66){
			printf("gpu 66 (x,x,x): %d-%d-%d  %d\n", i, j, k, bid);
			printf("gpu 66 (x,x, x): %f-%f-%f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
			}*/

			//assert(isfinite(tempPosition[0]));

			//assert(currentBlockIndex >= 0);
			resultPosition[posIdx] = tempPosition[0];
			resultPosition[posIdx + 1] = tempPosition[1];
			resultPosition[posIdx + 2] = tempPosition[2];

			//std::cout << "ijk: " << i << "-" << j << "-" << k << " idx: " << params->definedActiveBlock << std::endl;
		}
	}
}



__device__ __inline__ void rewind(float* sValues, unsigned int tid){

	while (tid < 11 * 11 * 11){
		const float buffer = sValues[tid];
		__syncthreads();
		if (tid > 0) sValues[tid - 1] = buffer;

		tid += blockDim.x;
	}
}

__device__ __inline__ unsigned long flatIdx(uint3 imgIdIn, const dim3 imgSize){
	assert(imgIdIn.x + imgIdIn.y *(imgSize.x) + imgIdIn.z * (imgSize.x * imgSize.y) < imgSize.x*imgSize.y*imgSize.z);
	return imgIdIn.x + imgIdIn.y *(imgSize.x) + imgIdIn.z * (imgSize.x * imgSize.y);
}

__device__ __inline__ unsigned long flatIdx(uint3 imgIdIn, uint3 imgSize){
	assert(imgIdIn.x + imgIdIn.y *(imgSize.x) + imgIdIn.z * (imgSize.x * imgSize.y) < imgSize.x*imgSize.y*imgSize.z);
	return imgIdIn.x + imgIdIn.y *(imgSize.x) + imgIdIn.z * (imgSize.x * imgSize.y);
}

__device__ __inline__ bool valid(uint3 imgIdIn, uint3 imgSize){
	return (imgIdIn.x >= 0 && imgIdIn.x < imgSize.x) && (imgIdIn.y >= 0 && imgIdIn.y < imgSize.y) && (imgIdIn.z >= 0 && imgIdIn.z < imgSize.z);
}

__device__ __inline__ uint3 idx3D(unsigned int tid, uint3 blockSize){
	const unsigned int xy = (blockSize.x * blockSize.y);
	const unsigned int idz = tid / xy;
	const unsigned int idy = (tid - xy * idz) / blockSize.x;
	const unsigned int idx = tid - xy * idz - blockSize.x * idy;
	/*assert(idx < blockSize.x);
	assert(idy < blockSize.y);
	assert(idz < blockSize.z);*/
	return make_uint3(idx, idy, idz);
}
//launched in 8*8*8 blocks (each block is double)
__global__ void preCompute(float* resultVar, float* resultTemp){

	//smem for intensity values plus halo
	__shared__ float sResultValues[11 * 11 * 11];

	float rResultVar, rResultTemp;
	unsigned int tid = threadIdx.x;//0-blockSize

	const uint3 tid3 = idx3D(tid, blockIdx);
	const uint3 imgBase = make_uint3(8 * blockIdx.x, 8 * blockIdx.y, 8 * blockIdx.z);//for now

	while (tid < 11 * 11 * 11){

		const uint3 id3In = idx3D(tid, make_uint3(11, 11, 11));
		const uint3 imageIdIn = make_uint3(imgBase.x + id3In.x, imgBase.y + id3In.y, imgBase.z + id3In.z);
		const unsigned long indexXYZIn = flatIdx(imageIdIn, c_ImageSize);

		if (valid(imageIdIn, c_ImageSize))
			sResultValues[tid] = tex1Dfetch(resultImageArray_texture, indexXYZIn);
		tid += blockDim.x;
	}
	__syncthreads();
	tid = threadIdx.x;

	const uint3 imgId = make_uint3(imgBase.x + tid3.x, imgBase.y + tid3.y, imgBase.z + tid3.z);
	unsigned long imgIdx = flatIdx(imgId, c_ImageSize);

	for (unsigned int k = 0; k < 4; k++)
	{
		for (unsigned int j = 0; j < 4; j++)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				const float rResultValue = sResultValues[tid];

				const float resultMean = reduceCustom_f1(rResultValue, tid, BLOCK_SIZE) / BLOCK_SIZE;
				const float resultTemp = rResultValue - resultMean;
				const float resultVar = reduceCustom_f1(resultTemp*resultTemp, tid, BLOCK_SIZE);
				/*int bx = 2 * blockIdx.x + ( tid3.x / 4);
				int by = 2 * blockIdx.y + (tid3.y / 4);
				int bz = 2 * blockIdx.z + (tid3.z / 4);
				if (bx == 8 && by == 0 && bz == 0 &&( i + j + k) == 0)  printf("pre- gpu 800 resultTemp: %f\n", resultTemp);
				if (bx == 8 && by == 0 && bz == 0 && (i + j + k) == 0)  printf("pre- gpu 800 resultVar: %f\n", resultVar);*/

				if (i == tid3.x % 4 && j == tid3.y % 4 && k == tid3.z % 4){
					rResultVar = resultVar;
					rResultTemp = resultTemp;

					/*if (bx == 8 && by == 0 && bz == 0 && i + j + k == 0)  printf("gpu 800 resultTemp: %f\n", resultTemp);
					if (bx == 8 && by == 0 && bz == 0 && i + j + k == 0)  printf("gpu 800 resultVar: %f\n", resultVar);*/
				}

				rewind(sResultValues, tid);
				__syncthreads();
			}
		}
	}
	if (valid(imgId, c_ImageSize)){
		resultTemp[imgIdx] = rResultTemp;
		resultVar[imgIdx] = rResultVar;
	}
}



__device__ __inline__ float countNans(float data, const unsigned int tid, bool condition){
	static __shared__ unsigned int sData2[64];

	sData2[tid] = isfinite(data) && condition ? 1 : 0;
	__syncthreads();
	/*const unsigned int laneId = tid % 32;
	const unsigned int warpid = tid / 32;*/


	/*assert(warpid == 0);
	assert(laneId == tid);
	assert(threadIdx.z<2);*/
	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

	/*else{
	assert(warpid == 1);
	assert(laneId != tid);
	assert(threadIdx.z>=2);
	}*/
	__syncthreads();
	return sData2[0];
}


//launched as 64 thread blocks
//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernel2pp(float *resultPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){

	__shared__ float sResultValues[12 * 12 * 12];


	const bool border = blockIdx.x == 0 || blockIdx.y == 0 || blockIdx.z == 0 || blockIdx.x >= gridDim.x - 2 || blockIdx.y >= gridDim.y - 2 || blockIdx.z >= gridDim.z - 2;

	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

	//bool is800 = (i == 22 && j == 9 && k == 1);
	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

	const unsigned int xBaseImage = blockIdx.x * 4;
	const unsigned int yBaseImage = blockIdx.y * 4;
	const unsigned int zBaseImage = blockIdx.z * 4;


	const unsigned int tid = threadIdx.x;//0-blockSize

	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;

	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
	//if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
	//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);

	for (int n = -1; n <= 1; n += 1)
	{
		for (int m = -1; m <= 1; m += 1)
		{
			for (int l = -1; l <= 1; l += 1)
			{
				const int x = l * 4 + idx;
				const int y = m * 4 + idy;
				const int z = n * 4 + idz;

				const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);

				const unsigned int xImageIn = xBaseImage + x;
				const unsigned int yImageIn = yBaseImage + y;
				const unsigned int zImageIn = zBaseImage + z;

				const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

				const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
				sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");
				//if (is800 && tid == 0 && l == 0 && m == 0 && n == 0) printf("s1 tid: %d | ResultValues: %f | sid: %d\n", tid, sResultValues[sIdx], sIdx);
			}
		}
	}

	const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");
	float targetMean, targetTemp, targetVar;

	if (!border){
		targetMean = REDUCE(rTargetValue, tid) / 64;
		targetTemp = rTargetValue - targetMean;
		targetVar = REDUCE(targetTemp*targetTemp, tid);
	}



	float bestDisplacement[3];
	float bestCC = 0.0f;

	// iteration over the result blocks
	for (unsigned int n = 1; n < 8; n += 1)
	{
		for (unsigned int m = 1; m < 8; m += 1)
		{
			for (unsigned int l = 1; l < 8; l += 1)
			{

				const unsigned int x = idx + l;
				const unsigned int y = idy + m;
				const unsigned int z = idz + n;

				const unsigned int sIdxIn = z * 12 * 12 + y * 12 + x;


				//bool condition1 = is800 && tid == 0 && l == 1 && m == 2 && n == 2;
				const float rResultValue = sResultValues[sIdxIn];
				bool finiteR = isfinite(rResultValue) && targetInBounds;

				const unsigned int bSize = border ? countNans(rResultValue, tid, targetInBounds) : 64;//out
				//if (is800 &&  l == 1 && m == 2 && n == 2) printf("tid: %d | sze: %d | RVL: %f | TIB: %d\n", tid, bSize, rResultValue, targetInBounds);
				//if (!border && bSize != 64) printf("(%d, %d, %d) BSZ: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bSize);
				if (bSize>32 && bSize <= 64){

					const float rChecked = finiteR ? rResultValue : 0.0f;
					const float tChecked = finiteR ? rTargetValue : 0.0f;
					if (border){
						targetMean = REDUCE(tChecked, tid) / bSize;//out
						targetTemp = finiteR ? rTargetValue - targetMean : 0.0f;
						targetVar = REDUCE(targetTemp*targetTemp, tid);//out
					}


					const float resultMean = REDUCE(rChecked, tid) / bSize;//out
					const float resultTemp = finiteR ? rResultValue - resultMean : 0.0f;
					const float resultVar = REDUCE(resultTemp*resultTemp, tid/*, is800 && l == 7 && m == 6 && n == 3*/);//out

					const float sumTargetResult = REDUCE((targetTemp)*(resultTemp), tid);//out
					const float localCC = fabs((sumTargetResult) / sqrtf(targetVar*resultVar));//out
					//if (condition1) printf("gpu 800 | sze: %d |TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, rTargetMean, rTargetVar, resultMean, resultVar, sumTargetResult, localCC);
					//if (condition1) printf("gpu 800 | RVL: %f | TVL: %f\n", rResultValue, rTargetValue);
					//__syncthreads();

					//warp vote here
					if (tid == 0 && localCC > bestCC) {
						bestCC = localCC;
						bestDisplacement[0] = l - 4.0f;
						bestDisplacement[1] = m - 4.0f;
						bestDisplacement[2] = n - 4.0f;
					}

				}
				//if (is800 && localCC > 0.99f) printf("%d-%d-%d: %f\n", l, m, n, localCC);
			}
		}
	}
	//if (is800 && tid == 0) printf("gpu 800  disp: %f - %f - %f | bestCC: %f\n", bestDisplacement[0], bestDisplacement[1], bestDisplacement[2], bestCC);
	//__syncthreads();
	if (tid == 0) {
		//if (is800) printf("gpu (%d, %d, %d): %d-%d: (%f, %f, %f)\n", i, j, k, bid, currentBlockIndex, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);

		//if (currentBlockIndex == 1635 / 3) printf("454: %d-%d-%d\n", i, j, k);
		float  tempPosition[3];

		bestDisplacement[0] += (blockIdx.x*BLOCK_WIDTH);
		bestDisplacement[1] += (blockIdx.y*BLOCK_WIDTH);
		bestDisplacement[2] += (blockIdx.z*BLOCK_WIDTH);

		reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);

		//if (is800) printf("gpu (8,0, 0): %f - %f - %f\n", tempPosition[0], tempPosition[1], tempPosition[2]);

		const unsigned int posIdx = 3 * currentBlockIndex;

		resultPosition[posIdx] = tempPosition[0];
		resultPosition[posIdx + 1] = tempPosition[1];
		resultPosition[posIdx + 2] = tempPosition[2];
	}
	//}
}

//launched as 64 thread blocks
//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernel2pp2(float *resultPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){

	__shared__ float sResultValues[12 * 12 * 12];


	const bool border = blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1 || blockIdx.z == gridDim.z - 1;

	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

	bool is800 = (blockIdx.x == 14 && blockIdx.y == 5 && blockIdx.z == 0);
	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

	const unsigned int xBaseImage = blockIdx.x * 4;
	const unsigned int yBaseImage = blockIdx.y * 4;
	const unsigned int zBaseImage = blockIdx.z * 4;


	const unsigned int tid = threadIdx.x;//0-blockSize

	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;

	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
	//if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
	//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);

	for (int n = -1; n <= 1; n += 1)
	{
		for (int m = -1; m <= 1; m += 1)
		{
			for (int l = -1; l <= 1; l += 1)
			{
				const int x = l * 4 + idx;
				const int y = m * 4 + idy;
				const int z = n * 4 + idz;

				const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);

				const unsigned int xImageIn = xBaseImage + x;
				const unsigned int yImageIn = yBaseImage + y;
				const unsigned int zImageIn = zBaseImage + z;

				const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

				const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
				sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");
				//if (is800 && tid == 0 && l == 0 && m == 0 && n == 0) printf("s1 tid: %d | ResultValues: %f | sid: %d\n", tid, sResultValues[sIdx], sIdx);
			}
		}
	}

	const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");

	const float targetMean = REDUCE(rTargetValue, tid) / 64;
	const float targetTemp = rTargetValue - targetMean;
	const float targetVar = REDUCE(targetTemp*targetTemp, tid);

	float bestDisplacement[3];
	float bestCC = 0.0f;

	// iteration over the result blocks
	for (unsigned int n = 1; n < 8; n += 1)
	{
		bool nBorder = n < 4 && blockIdx.z == 0 || n>4 && blockIdx.z >= gridDim.z - 2;
		for (unsigned int m = 1; m < 8; m += 1)
		{
			bool mBorder = m < 4 && blockIdx.y == 0 || m>4 && blockIdx.y >= gridDim.y - 2;
			for (unsigned int l = 1; l < 8; l += 1)
			{
				bool lBorder = l < 4 && blockIdx.x == 0 || l>4 && blockIdx.x >= gridDim.x - 2;

				const unsigned int x = idx + l;
				const unsigned int y = idy + m;
				const unsigned int z = idz + n;

				const unsigned int sIdxIn = z * 12 * 12 + y * 12 + x;

				/*bool neighbourIs = l == 2 && m == 6 && n == 3;
				bool condition1 = is800 && tid == 0 && neighbourIs;*/
				const float rResultValue = sResultValues[sIdxIn];
				bool overlap = isfinite(rResultValue) && targetInBounds;

				/*if (neighbourIs && is800 ) printf("rVal: %f | in: %d |tid: %d | trg: %f\n", rResultValue, targetInBounds, tid, rTargetValue);*/
				//if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n", l - 4, m - 4, n - 4, rResultValue, rTargetValue);
				const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
				//if (is800 &&  l == 6 && m == 6 && n == 6) printf("tid: %d | sze: %d | RVL: %f | TIB: %d\n", tid, bSize, rResultValue, targetInBounds);
				//if (!(nBorder || mBorder || lBorder || border) && bSize != 64) printf("(%d, %d, %d) BSZ: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bSize);
				//if (condition1) printf("sze: %d\n", bSize);

				if (bSize > 32 && bSize <= 64){

					const float rChecked = overlap ? rResultValue : 0.0f;
					float newTargetTemp = targetTemp;
					float ttargetvar = targetVar;
					if (bSize < 64){
						//if (condition1) printf("in bSize<64\n");
						const float tChecked = overlap ? rTargetValue : 0.0f;
						const float ttargetMean = REDUCE(tChecked, tid) / bSize;//out
						newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
						//if (neighbourIs && is800) printf("tmp: %f | ovp: %d |tid: %d \n", newTargetTemp, overlap, tid);
						ttargetvar = REDUCE(newTargetTemp*newTargetTemp, tid);//out
					}

					const float resultMean = REDUCE(rChecked, tid) / bSize;//out
					const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
					const float resultVar = REDUCE(resultTemp*resultTemp, tid/*, is800 && l == 7 && m == 6 && n == 3*/);//out

					const float sumTargetResult = REDUCE((newTargetTemp)*(resultTemp), tid);//out
					const float localCC = fabs((sumTargetResult) / sqrtf(ttargetvar*resultVar));//out

					/*if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n",l-4, m-4, n-4, rResultValue, rTargetValue);
					if (condition1) printf("sze: %d |TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);*/
					//__syncthreads();

					//warp vote here
					if (tid == 0 && localCC > bestCC) {
						bestCC = localCC;
						bestDisplacement[0] = l - 4.0f;
						bestDisplacement[1] = m - 4.0f;
						bestDisplacement[2] = n - 4.0f;
					}

				}
				//if (is800 && localCC > 0.99f) printf("%d-%d-%d: %f\n", l, m, n, localCC);
			}
		}
	}
	//if (is800 && tid == 0) printf("gpu 800  disp: %f - %f - %f | bestCC: %f\n", bestDisplacement[0], bestDisplacement[1], bestDisplacement[2], bestCC);
	//__syncthreads();
	if (tid == 0) {
		/*if (is800) printf("gpu (%d, %d, %d): %d-%d: (%f::%f::%f)\n", blockIdx.x, blockIdx.y, blockIdx.z, bid, currentBlockIndex, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);

		if (currentBlockIndex == 175 / 3) printf("175/3: %d-%d-%d\n", blockIdx.x, blockIdx.y, blockIdx.z);*/



		float  tempPosition[3];
		const unsigned int posIdx = 3 * currentBlockIndex;



		bestDisplacement[0] += (blockIdx.x*BLOCK_WIDTH);
		bestDisplacement[1] += (blockIdx.y*BLOCK_WIDTH);
		bestDisplacement[2] += (blockIdx.z*BLOCK_WIDTH);

		reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);

		//if (is800) printf("gpu (8,0, 0): %f - %f - %f\n", tempPosition[0], tempPosition[1], tempPosition[2]);



		resultPosition[posIdx] = tempPosition[0];
		resultPosition[posIdx + 1] = tempPosition[1];
		resultPosition[posIdx + 2] = tempPosition[2];
	}
	//}
}

//__global__ void resultsKernel2pp21(float *resultPosition, float *targetPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){
//
//	__shared__ float sResultValues[12 * 12 * 12];
//
//
//	const bool border = blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1 || blockIdx.z == gridDim.z - 1;
//
//	const unsigned int idz = threadIdx.x / 16;
//	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
//	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;
//
//	bool is800 = (blockIdx.x == 14 && blockIdx.y == 5 && blockIdx.z == 0);
//	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;
//
//	const unsigned int xBaseImage = blockIdx.x * 4;
//	const unsigned int yBaseImage = blockIdx.y * 4;
//	const unsigned int zBaseImage = blockIdx.z * 4;
//
//
//	const unsigned int tid = threadIdx.x;//0-blockSize
//
//	const unsigned int xImage = xBaseImage + idx;
//	const unsigned int yImage = yBaseImage + idy;
//	const unsigned int zImage = zBaseImage + idz;
//
//	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
//	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;
//
//	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
//	//if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
//	//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);
//
//	for (int n = -1; n <= 1; n += 1)
//	{
//		for (int m = -1; m <= 1; m += 1)
//		{
//			for (int l = -1; l <= 1; l += 1)
//			{
//				const int x = l * 4 + idx;
//				const int y = m * 4 + idy;
//				const int z = n * 4 + idz;
//
//				const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);
//
//				const unsigned int xImageIn = xBaseImage + x;
//				const unsigned int yImageIn = yBaseImage + y;
//				const unsigned int zImageIn = zBaseImage + z;
//
//				const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);
//
//				const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
//				sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");
//				//if (is800 && tid == 0 && l == 0 && m == 0 && n == 0) printf("s1 tid: %d | ResultValues: %f | sid: %d\n", tid, sResultValues[sIdx], sIdx);
//			}
//		}
//	}
//
//	const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");
//
//	const float targetMean = REDUCE(rTargetValue, tid) / 64;
//	const float targetTemp = rTargetValue - targetMean;
//	const float targetVar = REDUCE(targetTemp*targetTemp, tid);
//
//	float bestDisplacement[3];
//	float bestCC = 0.0f;
//
//	// iteration over the result blocks
//	for (unsigned int n = 1; n < 8; n += 1)
//	{
//		bool nBorder = n < 4 && blockIdx.z == 0 || n>4 && blockIdx.z >= gridDim.z - 2;
//		for (unsigned int m = 1; m < 8; m += 1)
//		{
//			bool mBorder = m < 4 && blockIdx.y == 0 || m>4 && blockIdx.y >= gridDim.y - 2;
//			for (unsigned int l = 1; l < 8; l += 1)
//			{
//				bool lBorder = l < 4 && blockIdx.x == 0 || l>4 && blockIdx.x >= gridDim.x - 2;
//
//				const unsigned int x = idx + l;
//				const unsigned int y = idy + m;
//				const unsigned int z = idz + n;
//
//				const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;
//
//				/*bool neighbourIs = l == 2 && m == 6 && n == 3;
//				bool condition1 = is800 && tid == 0 && neighbourIs;*/
//				const float rResultValue = sResultValues[sIdxIn];
//				bool overlap = isfinite(rResultValue) && targetInBounds;
//
//				/*if (neighbourIs && is800 ) printf("rVal: %f | in: %d |tid: %d | trg: %f\n", rResultValue, targetInBounds, tid, rTargetValue);*/
//				//if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n", l - 4, m - 4, n - 4, rResultValue, rTargetValue);
//				const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
//				//if (is800 &&  l == 6 && m == 6 && n == 6) printf("tid: %d | sze: %d | RVL: %f | TIB: %d\n", tid, bSize, rResultValue, targetInBounds);
//				//if (!(nBorder || mBorder || lBorder || border) && bSize != 64) printf("(%d, %d, %d) BSZ: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bSize);
//				//if (condition1) printf("sze: %d\n", bSize);
//
//				if (bSize > 32 && bSize <= 64){
//
//					const float rChecked = overlap ? rResultValue : 0.0f;
//					float newTargetTemp = targetTemp;
//					float ttargetvar = targetVar;
//					if (bSize < 64){
//						//if (condition1) printf("in bSize<64\n");
//						const float tChecked = overlap ? rTargetValue : 0.0f;
//						const float ttargetMean = REDUCE(tChecked, tid) / bSize;//out
//						newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
//						//if (neighbourIs && is800) printf("tmp: %f | ovp: %d |tid: %d \n", newTargetTemp, overlap, tid);
//						ttargetvar = REDUCE(newTargetTemp*newTargetTemp, tid);//out
//					}
//
//					const float resultMean = REDUCE(rChecked, tid) / bSize;//out
//					const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
//					const float resultVar = REDUCE(resultTemp*resultTemp, tid/*, is800 && l == 7 && m == 6 && n == 3*/);//out
//
//					const float sumTargetResult = REDUCE((newTargetTemp)*(resultTemp), tid);//out
//					const float localCC = fabs((sumTargetResult) / sqrtf(ttargetvar*resultVar));//out
//
//					/*if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n",l-4, m-4, n-4, rResultValue, rTargetValue);
//					if (condition1) printf("sze: %d |TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);*/
//					//__syncthreads();
//
//					//warp vote here
//					if (tid == 0 && localCC > bestCC) {
//						bestCC = localCC;
//						bestDisplacement[0] = l - 4.0f;
//						bestDisplacement[1] = m - 4.0f;
//						bestDisplacement[2] = n - 4.0f;
//					}
//
//				}
//				//if (is800 && localCC > 0.99f) printf("%d-%d-%d: %f\n", l, m, n, localCC);
//			}
//		}
//	}
//	//if (is800 && tid == 0) printf("gpu 800  disp: %f - %f - %f | bestCC: %f\n", bestDisplacement[0], bestDisplacement[1], bestDisplacement[2], bestCC);
//	//__syncthreads();
//	if (tid == 0) {
//		/*if (is800) printf("gpu (%d, %d, %d): %d-%d: (%f::%f::%f)\n", blockIdx.x, blockIdx.y, blockIdx.z, bid, currentBlockIndex, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);
//
//		if (currentBlockIndex == 175 / 3) printf("175/3: %d-%d-%d\n", blockIdx.x, blockIdx.y, blockIdx.z);*/
//
//
//		const unsigned int posIdx = 3 * currentBlockIndex;
//		resultPosition += posIdx;
//		targetPosition += posIdx;
//		float  targetPosition_temp[3];
//		targetPosition_temp[0] = (blockIdx.x*BLOCK_WIDTH);
//		targetPosition_temp[1] = (blockIdx.y*BLOCK_WIDTH);
//		targetPosition_temp[2] = (blockIdx.z*BLOCK_WIDTH);
//
//		bestDisplacement[0] += targetPosition_temp[0];
//		bestDisplacement[1] += targetPosition_temp[1];
//		bestDisplacement[2] += targetPosition_temp[2];
//
//
//		//float  tempPosition[3];
//		reg_mat44_mul_cuda(targetMatrix_xyz, targetPosition_temp, targetPosition);
//		reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, resultPosition);
//
//		/*targetPosition[posIdx] = tempPosition[0];
//		targetPosition[posIdx + 1] = tempPosition[1];
//		targetPosition[posIdx + 2] = tempPosition[2];*/
//
//
//
//
//
//
//
//		//if (is800) printf("gpu (8,0, 0): %f - %f - %f\n", tempPosition[0], tempPosition[1], tempPosition[2]);
//
//
//
//		/*resultPosition[posIdx] = tempPosition[0];
//		resultPosition[posIdx + 1] = tempPosition[1];
//		resultPosition[posIdx + 2] = tempPosition[2];*/
//	}
//	//}
//}

//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernel2pp21(float *resultPosition, float *targetPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims, unsigned int* definedBlock){

	__shared__ float sResultValues[12 * 12 * 12];


	const bool border = blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1 || blockIdx.z == gridDim.z - 1;

	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

	const unsigned int xBaseImage = blockIdx.x * 4;
	const unsigned int yBaseImage = blockIdx.y * 4;
	const unsigned int zBaseImage = blockIdx.z * 4;


	const unsigned int tid = threadIdx.x;//0-blockSize

	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;

	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);

	if (currentBlockIndex > -1){

		for (int n = -1; n <= 1; n += 1)
		{
			for (int m = -1; m <= 1; m += 1)
			{
				for (int l = -1; l <= 1; l += 1)
				{
					const int x = l * 4 + idx;
					const int y = m * 4 + idy;
					const int z = n * 4 + idz;

					const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);

					const unsigned int xImageIn = xBaseImage + x;
					const unsigned int yImageIn = yBaseImage + y;
					const unsigned int zImageIn = zBaseImage + z;

					const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
					sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");

				}
			}
		}

		const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");

		const float targetMean = REDUCE(rTargetValue, tid) / 64;
		const float targetTemp = rTargetValue - targetMean;
		const float targetVar = REDUCE(targetTemp*targetTemp, tid);

		float bestDisplacement[3] = { nanf("sNaN"), nanf("sNaN"), nanf("sNaN") };
		float bestCC = 0.0f;

		// iteration over the result blocks
		for (unsigned int n = 1; n < 8; n += 1)
		{
			bool nBorder = n < 4 && blockIdx.z == 0 || n>4 && blockIdx.z >= gridDim.z - 2;
			for (unsigned int m = 1; m < 8; m += 1)
			{
				bool mBorder = m < 4 && blockIdx.y == 0 || m>4 && blockIdx.y >= gridDim.y - 2;
				for (unsigned int l = 1; l < 8; l += 1)
				{
					bool lBorder = l < 4 && blockIdx.x == 0 || l>4 && blockIdx.x >= gridDim.x - 2;

					const unsigned int x = idx + l;
					const unsigned int y = idy + m;
					const unsigned int z = idz + n;

					const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;

					const float rResultValue = sResultValues[sIdxIn];
					bool overlap = isfinite(rResultValue) && targetInBounds;
					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out


					if (bSize > 32 && bSize <= 64){

						const float rChecked = overlap ? rResultValue : 0.0f;
						float newTargetTemp = targetTemp;
						float ttargetvar = targetVar;
						if (bSize < 64){

							const float tChecked = overlap ? rTargetValue : 0.0f;
							const float ttargetMean = REDUCE(tChecked, tid) / bSize;
							newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
							ttargetvar = REDUCE(newTargetTemp*newTargetTemp, tid);
						}

						const float resultMean = REDUCE(rChecked, tid) / bSize;
						const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
						const float resultVar = REDUCE(resultTemp*resultTemp, tid);

						const float sumTargetResult = REDUCE((newTargetTemp)*(resultTemp), tid);
						const float localCC = fabs((sumTargetResult) / sqrtf(ttargetvar*resultVar));


						//warp vote here
						if (tid == 0 && localCC > bestCC) {
							bestCC = localCC;
							bestDisplacement[0] = l - 4.0f;
							bestDisplacement[1] = m - 4.0f;
							bestDisplacement[2] = n - 4.0f;
						}

					}
				}
			}
		}

		if (tid == 0 && isfinite(bestDisplacement[0])) {

			//const unsigned int posIdx = 3 * currentBlockIndex;
			const unsigned int posIdx = 3 * atomicAdd(&(definedBlock[0]), 1);
			//printf("%d: %d \n", definedBlock[0], bid);
			resultPosition += posIdx;
			targetPosition += posIdx;
			float  targetPosition_temp[3];
			targetPosition_temp[0] = (blockIdx.x*BLOCK_WIDTH);
			targetPosition_temp[1] = (blockIdx.y*BLOCK_WIDTH);
			targetPosition_temp[2] = (blockIdx.z*BLOCK_WIDTH);

			bestDisplacement[0] += targetPosition_temp[0];
			bestDisplacement[1] += targetPosition_temp[1];
			bestDisplacement[2] += targetPosition_temp[2];


			//float  tempPosition[3];
			reg_mat44_mul_cuda(targetMatrix_xyz, targetPosition_temp, targetPosition);
			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, resultPosition);
		}
		//else if (tid == 0 && !isfinite(bestDisplacement[0])){
		//	const unsigned int posIdx = 3 * currentBlockIndex;
		//	/*const unsigned int posIdx = 3 * atomicAdd(&(definedBlock[0]), 1);*/
		//	//printf("%d: %d \n", definedBlock[0], bid);
		//	resultPosition += posIdx;
		//	targetPosition += posIdx;

		//	resultPosition[0] = -10000.0f;
		//	resultPosition[1] = -10000.0f;
		//	resultPosition[2] = -10000.0f;

		//	targetPosition[0] = 0.0f;
		//	targetPosition[1] = 0.0f;
		//	targetPosition[2] =0.0f;
		//}
	}

}


//launched as 64 thread blocks
//Blocks: 1-(n-1) for all dimensions
__global__ void resultsKernelNoBC(float *resultPosition, int* mask, float* targetMatrix_xyz, uint3 blockDims){

	__shared__ float sResultValues[12 * 12 * 12];


	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

	//bool is800 = (blockIdx.x == 14 && blockIdx.y == 5 && blockIdx.z == 0);
	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

	const unsigned int xBaseImage = blockIdx.x * 4;
	const unsigned int yBaseImage = blockIdx.y * 4;
	const unsigned int zBaseImage = blockIdx.z * 4;


	const unsigned int tid = threadIdx.x;//0-blockSize

	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;

	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);
	//if (currentBlockIndex >= 0 && xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z){
	//if (i == 22 && j == 10 && k == 0) printf("size: %d | idx: %d | flat: %lu | size: %lu \n", imageSize.x, xImage, indexXYZ, imageSize.x*imageSize.y*imageSize.z);

	for (int n = -1; n <= 1; n += 1)
	{
		for (int m = -1; m <= 1; m += 1)
		{
			for (int l = -1; l <= 1; l += 1)
			{
				const int x = l * 4 + idx;
				const int y = m * 4 + idy;
				const int z = n * 4 + idz;

				const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);

				const unsigned int xImageIn = xBaseImage + x;
				const unsigned int yImageIn = yBaseImage + y;
				const unsigned int zImageIn = zBaseImage + z;

				const unsigned int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

				const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
				sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");
				//if (is800 && tid == 0 && l == 0 && m == 0 && n == 0) printf("s1 tid: %d | ResultValues: %f | sid: %d\n", tid, sResultValues[sIdx], sIdx);
			}
		}
	}

	const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");

	const float targetMean = REDUCE(rTargetValue, tid) / 64;
	const float targetTemp = rTargetValue - targetMean;
	const float targetVar = REDUCE(targetTemp*targetTemp, tid);

	float bestDisplacement[3];
	float bestCC = 0.0f;

	// iteration over the result blocks
	for (unsigned int n = 1; n < 8; n += 1)
	{
		bool nBorder = n < 4 && blockIdx.z == 0 || n>4 && blockIdx.z >= gridDim.z - 2;
		for (unsigned int m = 1; m < 8; m += 1)
		{
			bool mBorder = m < 4 && blockIdx.y == 0 || m>4 && blockIdx.y >= gridDim.y - 2;
			for (unsigned int l = 1; l < 8; l += 1)
			{
				bool lBorder = l < 4 && blockIdx.x == 0 || l>4 && blockIdx.x >= gridDim.x - 2;

				const unsigned int x = idx + l;
				const unsigned int y = idy + m;
				const unsigned int z = idz + n;

				const unsigned int sIdxIn = z * 12 * 12 + y * 12 + x;

				/*bool neighbourIs = l == 2 && m == 6 && n == 3;
				bool condition1 = is800 && tid == 0 && neighbourIs;*/
				const float rResultValue = sResultValues[sIdxIn];
				//bool overlap = isfinite(rResultValue) && targetInBounds;
				const unsigned int bSize = 64;
				/*if (neighbourIs && is800 ) printf("rVal: %f | in: %d |tid: %d | trg: %f\n", rResultValue, targetInBounds, tid, rTargetValue);*/
				//if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n", l - 4, m - 4, n - 4, rResultValue, rTargetValue);
				//const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
				//if (is800 &&  l == 6 && m == 6 && n == 6) printf("tid: %d | sze: %d | RVL: %f | TIB: %d\n", tid, bSize, rResultValue, targetInBounds);
				//if (!(nBorder || mBorder || lBorder || border) && bSize != 64) printf("(%d, %d, %d) BSZ: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bSize);
				//if (condition1) printf("sze: %d\n", bSize);


				const float resultMean = REDUCE(rResultValue, tid) / bSize;//out
				const float resultTemp = rResultValue - resultMean;
				const float resultVar = REDUCE(resultTemp*resultTemp, tid/*, is800 && l == 7 && m == 6 && n == 3*/);//out

				const float sumTargetResult = REDUCE((rTargetValue)*(resultTemp), tid);//out
				const float localCC = fabs((sumTargetResult) / sqrtf(targetVar*resultVar));//out

				/*if (condition1) printf("gpu %d::%d::%d | RVL: %f | TVL: %f\n",l-4, m-4, n-4, rResultValue, rTargetValue);
				if (condition1) printf("sze: %d |TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);*/
				//__syncthreads();

				//warp vote here
				if (tid == 0 && localCC > bestCC) {
					bestCC = localCC;
					bestDisplacement[0] = l - 4.0f;
					bestDisplacement[1] = m - 4.0f;
					bestDisplacement[2] = n - 4.0f;
				}

			}
			//if (is800 && localCC > 0.99f) printf("%d-%d-%d: %f\n", l, m, n, localCC);
		}
	}

	//if (is800 && tid == 0) printf("gpu 800  disp: %f - %f - %f | bestCC: %f\n", bestDisplacement[0], bestDisplacement[1], bestDisplacement[2], bestCC);
	//__syncthreads();
	if (tid == 0) {
		/*if (is800) printf("gpu (%d, %d, %d): %d-%d: (%f::%f::%f)\n", blockIdx.x, blockIdx.y, blockIdx.z, bid, currentBlockIndex, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);

		if (currentBlockIndex == 175 / 3) printf("175/3: %d-%d-%d\n", blockIdx.x, blockIdx.y, blockIdx.z);*/
		float  tempPosition[3];

		bestDisplacement[0] += (blockIdx.x*BLOCK_WIDTH);
		bestDisplacement[1] += (blockIdx.y*BLOCK_WIDTH);
		bestDisplacement[2] += (blockIdx.z*BLOCK_WIDTH);

		reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, tempPosition);

		//if (is800) printf("gpu (8,0, 0): %f - %f - %f\n", tempPosition[0], tempPosition[1], tempPosition[2]);

		const unsigned int posIdx = 3 * currentBlockIndex;

		resultPosition[posIdx] = tempPosition[0];
		resultPosition[posIdx + 1] = tempPosition[1];
		resultPosition[posIdx + 2] = tempPosition[2];
	}
	//}
}

#endif
