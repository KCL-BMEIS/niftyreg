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

#define REDUCE reduceCustom
//#define REDUCE blockReduceSum

#include "assert.h"
#include "_reg_blockMatching.h"
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



texture<float, 1, cudaReadModeElementType> targetImageArray_texture;
texture<float, 1, cudaReadModeElementType> resultImageArray_texture;
texture<int, 1, cudaReadModeElementType> activeBlock_texture;






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

//Marc's kernel
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

//Marc's kernel
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

	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

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


__inline__ __device__
float blockReduceSum(float val, int tid) {

	static __shared__ float shared[2];
	int laneId = tid % 32;
	int warpId = tid / 32;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (laneId == 0) shared[warpId] = val;
	//if (blockIdx.x == 8 && blockIdx.y == 0 && blockIdx.z == 0) printf("idx: %d | lane: %d \n", tid, lane);
	__syncthreads();


	return shared[0] + shared[1];
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



__device__ __inline__ float countNans(float data, const unsigned int tid, bool condition){
	static __shared__ unsigned int sData2[64];

	sData2[tid] = isfinite(data) && condition ? 1 : 0;
	__syncthreads();


	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

	__syncthreads();
	return sData2[0];
}



__global__ void blockMatchingKernel(float *resultPosition, float *targetPosition, int* mask, float* targetMatrix_xyz, unsigned int* definedBlock, uint3 c_ImageSize){

	__shared__ float sResultValues[12 * 12 * 12];

	//const bool is_7_21_11 = blockIdx.x == 7 && blockIdx.y == 21 && blockIdx.z == 11;
//	bool b2_13_10 = blockIdx.x==2&&blockIdx.y==13&&blockIdx.z==10;
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

					const int xImageIn = xBaseImage + x;
					const int yImageIn = yBaseImage + y;
					const int zImageIn = zBaseImage + z;

					const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
					sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");

				}
			}
		}

		const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");

		const float targetMean = REDUCE(rTargetValue, tid) / 64;
		const float targetTemp = rTargetValue - targetMean;
		const float targetVar = REDUCE(targetTemp*targetTemp, tid);

		float bestDisplacement[3] = { nanf("sNaN"),0.0f,0.0f };
		float bestCC = 0.0f;

		// iteration over the result blocks
		for (unsigned int n = 1; n < 8; n += 1)
		{
			const bool nBorder = (n < 4 && blockIdx.z == 0) || (n>4 && blockIdx.z >= gridDim.z - 2);
			for (unsigned int m = 1; m < 8; m += 1)
			{
				const bool mBorder = (m < 4 && blockIdx.y == 0) || (m>4 && blockIdx.y >= gridDim.y - 2);
				for (unsigned int l = 1; l < 8; l += 1)
				{

					/*bool nIs_1_0_m3 = l == 1 + 4 && m == 0 + 4 && n == -3 + 4;
					bool nIs_1_0_m2 = l == 1 + 4 && m == 0 + 4 && n == -2 + 4;

					bool condition1 = b2_13_10  && nIs_1_0_m3 && tid==0;
					bool condition2 = b2_13_10  && nIs_1_0_m2 && tid==0;*/

					const bool lBorder = (l < 4 && blockIdx.x == 0) || (l>4 && blockIdx.x >= gridDim.x - 2);

					const unsigned int x = idx + l;
					const unsigned int y = idy + m;
					const unsigned int z = idz + n;

					const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;

					const float rResultValue = sResultValues[sIdxIn];
					const bool overlap = isfinite(rResultValue) && targetInBounds;
//					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? (unsigned int)REDUCE(overlap?1.0f:0.0f, tid) : 64;//out


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

						/*if (condition1) printf("GPU -3 | sze: %d | TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);
						if (condition2) printf("GPU -2 | sze: %d | TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);
*/
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

//			if (b2_13_10) printf("disp: %f-%f-%f\n", bestDisplacement[0],bestDisplacement[1], bestDisplacement[2]);
			const unsigned int posIdx = 3 * atomicAdd(&(definedBlock[0]), 1);
			//printf("%d: %d \n", definedBlock[0], bid);
			resultPosition += posIdx;
			targetPosition += posIdx;

			const float targetPosition_temp[3] = {blockIdx.x*BLOCK_WIDTH,blockIdx.y*BLOCK_WIDTH, blockIdx.z*BLOCK_WIDTH };

			bestDisplacement[0] += targetPosition_temp[0];
			bestDisplacement[1] += targetPosition_temp[1];
			bestDisplacement[2] += targetPosition_temp[2];

			//float  tempPosition[3];
			reg_mat44_mul_cuda(targetMatrix_xyz, targetPosition_temp, targetPosition);
			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, resultPosition);

		}
	}

}

#endif
