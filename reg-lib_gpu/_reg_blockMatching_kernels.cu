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

// Some parameters that we need for the kernel execution.
// The caller is supposed to ensure that the values are set

// Number of blocks in each dimension
__device__ __constant__ int3 c_BlockDim;
__device__ __constant__ int c_StepSize;
__device__ __constant__ int3 c_ImageSize;
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

// CUDA kernel to process the target values
__global__ void process_target_blocks_gpu(float *targetPosition_d,                                          
                                          float *targetValues)                                          
{   
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * gridDim.x);
    const int3 bDim = c_BlockDim;
	if (tid < bDim.x * bDim.y * bDim.z){
        const int currentBlockIndex = tex1Dfetch(activeBlock_texture,tid);        
        if (currentBlockIndex >= 0){
	        // Get the corresponding (i, j, k) indices	        
            int tempIndex = currentBlockIndex;
	        const int k =(int)(tempIndex/(bDim.x * bDim.y));
	        tempIndex -= k * bDim.x * bDim.y;
	        const int j =(int)(tempIndex/(bDim.x));
	        const int i = tempIndex - j * (bDim.x);
            const int offset = tid * BLOCK_SIZE;
		    const int targetIndex_start_x = i * BLOCK_WIDTH;
		    const int targetIndex_start_y = j * BLOCK_WIDTH;
		    const int targetIndex_start_z = k * BLOCK_WIDTH;
    		
		    int targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;
		    int targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;
		    int targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;

            const int3 imageSize = c_ImageSize;
            const int threadId = tid % NUM_BLOCKS_TO_COMPARE;
            if (targetIndex_end_z > imageSize.z){                
                targetIndex_end_z = imageSize.z;
            }

            __shared__ int rampY[Block_target_block];
            rampY[threadId] = 0;
            if (targetIndex_end_y > imageSize.y){
                rampY[threadId] = targetIndex_end_y - imageSize.y;
                targetIndex_end_y = imageSize.y;
            }
            
            __shared__ int rampX[Block_target_block];
            rampX[threadId] = 0;
            if (targetIndex_end_x > imageSize.x){
                rampX[threadId] = targetIndex_end_x - imageSize.x;
                targetIndex_end_x = imageSize.x;
            }

            for (int count = 0; count < BLOCK_SIZE; ++count)
            {
                targetValues[count + offset] = 0.0f;
            }

		    unsigned int index = 0;	            
		    for(int z = targetIndex_start_z; z< targetIndex_end_z; ++z){                
		        int indexZ = z * imageSize.x * imageSize.y;
		        for(int y = targetIndex_start_y; y < targetIndex_end_y; ++y){                    
			        int indexXYZ = indexZ + y * imageSize.x + targetIndex_start_x;
                    for(int x = targetIndex_start_x; x < targetIndex_end_x; ++x){
				        targetValues[index + offset] = tex1Dfetch(targetImageArray_texture, indexXYZ);
			            indexXYZ++;
			            index++;
			        }
                    index += rampX[threadId];
		        }
                index += rampY[threadId] * BLOCK_WIDTH;
		    }

            float4 targetPosition;
		    targetPosition.x = i * BLOCK_WIDTH;
		    targetPosition.y = j * BLOCK_WIDTH;
		    targetPosition.z = k * BLOCK_WIDTH;        		    
            apply_affine(targetPosition, &(targetPosition_d[tid * 3]));            
        }
    }
}


// CUDA kernel to process the result blocks
__global__ void process_result_blocks_gpu(float *resultPosition_d,
                                          float *targetValues)
{    
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * gridDim.x);    
    const int3 bDim = c_BlockDim;
    const int ctid = (int)(tid / NUM_BLOCKS_TO_COMPARE);
    __shared__ float4 localCC [NUM_BLOCKS_TO_COMPARE];
    localCC[tid % NUM_BLOCKS_TO_COMPARE] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    __shared__ int updateThreadID;
    updateThreadID = -1;
    if (ctid < bDim.x * bDim.y * bDim.z) {
        const int activeBlockIndex = tex1Dfetch(activeBlock_texture, ctid);        
        int tempIndex = activeBlockIndex;
        const int k =(int)(tempIndex/(bDim.x * bDim.y));
	    tempIndex -= k * bDim.x * bDim.y;
	    const int j =(int)(tempIndex/(bDim.x));
	    const int i = tempIndex - j * (bDim.x);		        
		const int targetIndex_start_x = i * BLOCK_WIDTH;
		const int targetIndex_start_y = j * BLOCK_WIDTH;
		const int targetIndex_start_z = k * BLOCK_WIDTH;			
        
        if (activeBlockIndex >= 0) {            
            const int block_offset = ctid * BLOCK_SIZE;
            const int3 imageSize = c_ImageSize;
            tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
            int n = (int)tempIndex /NUM_BLOCKS_TO_COMPARE_2D;
            tempIndex -= n * NUM_BLOCKS_TO_COMPARE_2D;            
            int m = (int)tempIndex /NUM_BLOCKS_TO_COMPARE_1D;
            int l = tempIndex - m * NUM_BLOCKS_TO_COMPARE_1D;
            n -= OVERLAP_SIZE;
            m -= OVERLAP_SIZE;
            l -= OVERLAP_SIZE;

            tempIndex = tid % NUM_BLOCKS_TO_COMPARE;

            int resultIndex_start_z = targetIndex_start_z + n;
            int resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;

            int rampZ = 0;            
            if (resultIndex_start_z < 0){
                rampZ = -resultIndex_start_z;
                resultIndex_start_z = 0;                    
            }
            if (resultIndex_end_z > imageSize.z){
                resultIndex_end_z = imageSize.z;
            }

            int resultIndex_start_y = targetIndex_start_y + m;
			int resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
            int rampYLeft = 0;
            int rampYRight = 0;
            if (resultIndex_start_y < 0) {
                rampYLeft = -resultIndex_start_y;
                resultIndex_start_y = 0;
            }

            if (resultIndex_end_y > imageSize.y) {
                rampYRight = resultIndex_end_y - imageSize.y;
                resultIndex_end_y = imageSize.y;
            }

            int resultIndex_start_x = targetIndex_start_x + l;
			int resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
            int rampXLeft = 0;
            int rampXRight = 0;
            if (resultIndex_start_x < 0) {
                rampXLeft = -resultIndex_start_x;
                resultIndex_start_x = 0;
            }
            if (resultIndex_end_x > imageSize.x) {
                rampXRight = resultIndex_end_x - imageSize.x;
                resultIndex_end_x = imageSize.x;
            }            

            float target_mean = 0.0f;            
            float result_mean = 0.0f;            
            float voxel_number = 0.0f;
            float result_var = 0.0f;
            float target_var = 0.0f;
            float target_temp = 0.0f;
            float result_temp = 0.0f;
            float current_value = 0.0f;
            float current_target_value = 0.0f;

            localCC[tempIndex].w = 0.0f;
            __shared__ unsigned int index[NUM_BLOCKS_TO_COMPARE];
            index[tempIndex]= rampZ * BLOCK_WIDTH * BLOCK_WIDTH;
            for(int z = resultIndex_start_z; z< resultIndex_end_z; ++z){
                int indexZ = z * imageSize.y * imageSize.x;
                index[tempIndex] += rampYLeft * BLOCK_WIDTH;
                for(int y = resultIndex_start_y; y < resultIndex_end_y; ++y){                                    
                    int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
                    index[tempIndex] += rampXLeft;                                
                    for(int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
                        current_value = tex1Dfetch(resultImageArray_texture, indexXYZ);
                        current_target_value = targetValues[block_offset + index[tempIndex]];
                        if (current_value != 0.0f && current_target_value != 0.0f) {
                            result_mean += current_value;
                            target_mean += current_target_value;
                            ++voxel_number;
                        }
                        indexXYZ++;
                        index[tempIndex]++;
                    }
                    index[tempIndex] += rampXRight;                                    
                }
                index[tempIndex] += rampYRight * BLOCK_WIDTH;                            
            }

            if (voxel_number > 0.0f) {
                result_mean /= voxel_number;
                target_mean /= voxel_number;
            }

            index[tempIndex] = rampZ * BLOCK_WIDTH * BLOCK_WIDTH;
            for(int z = resultIndex_start_z; z< resultIndex_end_z; ++z){
                int indexZ = z * imageSize.y * imageSize.x;
                index[tempIndex] += rampYLeft * BLOCK_WIDTH;
                for(int y = resultIndex_start_y; y < resultIndex_end_y; ++y){                                    
                    int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
                    index[tempIndex] += rampXLeft;                                
                    for(int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
                        current_value = tex1Dfetch(resultImageArray_texture, indexXYZ);
                        current_target_value = targetValues[block_offset + index[tempIndex]];
                        if (current_value != 0.0f && current_target_value != 0.0f) {
                            target_temp = (current_target_value - target_mean);
                            result_temp = (current_value - result_mean);
                            result_var += result_temp * result_temp;
                            target_var += target_temp * target_temp;
                            localCC[tempIndex].w += target_temp * result_temp;
                        }
                        indexXYZ++;
                        index[tempIndex]++;
                    }
                    index[tempIndex] += rampXRight;                                    
                }
                index[tempIndex] += rampYRight * BLOCK_WIDTH;                            
            }            
            localCC[tempIndex].x = l;
            localCC[tempIndex].y = m;
            localCC[tempIndex].z = n;
            
            if (voxel_number > 0.0f) {
                target_var = sqrt(target_var/voxel_number);
                result_var = sqrt(result_var/voxel_number);
                if (target_var > 0.0f && result_var > 0.0f)
			        localCC[tempIndex].w = fabsf(localCC[tempIndex].w/
				        (voxel_number*target_var*result_var));
            }
            // Just take ownership of updating the final value
            if (updateThreadID == -1)
                updateThreadID = tid;
        }

        __syncthreads();
        // Just let one thread do the final update
        if (updateThreadID > -1) {
            float4 bestCC = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            __shared__ unsigned dex[NUM_BLOCKS_TO_COMPARE];
            for (dex[tempIndex] = 0; dex[tempIndex] < NUM_BLOCKS_TO_COMPARE; ++dex[tempIndex]) {
                if (localCC[dex[tempIndex]].w > bestCC.w) {
                    bestCC.x = localCC[dex[tempIndex]].x;
                    bestCC.y = localCC[dex[tempIndex]].y;
                    bestCC.z = localCC[dex[tempIndex]].z;
                    bestCC.w = localCC[dex[tempIndex]].w;
                }
            }
            bestCC.x += targetIndex_start_x;
            bestCC.y += targetIndex_start_y;
            bestCC.z += targetIndex_start_z;            
            apply_affine(bestCC, &(resultPosition_d[ctid * 3]));
        }
    }
}

#endif

