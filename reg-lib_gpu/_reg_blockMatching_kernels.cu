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
__device__ void apply_affine(const float3 &pt, float * result)
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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int3 bDim = c_BlockDim;
	if (tid < bDim.x * bDim.y * bDim.z){
        const int currentBlockIndex = tex1Dfetch(activeBlock_texture,tid);
        if (currentBlockIndex >= 0){
	        // Get the corresponding (i, j, k) indices
	        int tempIndex = tid;
	        const int k =(int)(tempIndex/(bDim.x * bDim.y));
	        tempIndex -= k * bDim.x * bDim.y;
	        const int j =(int)(tempIndex/(bDim.x));
	        const int i = tempIndex - j * (bDim.x);

		    const int offset = currentBlockIndex * BLOCK_SIZE;
		    const int targetIndex_start_x = i * BLOCK_WIDTH;
		    const int targetIndex_start_y = j * BLOCK_WIDTH;
		    const int targetIndex_start_z = k * BLOCK_WIDTH;
    		
		    int targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;
		    int targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;
		    int targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;

            const int3 imageSize = c_ImageSize;
            unsigned int index = 0;

            for(int z = targetIndex_start_z; z< targetIndex_end_z; ++z){
				if(-1<z && z<imageSize.z){
					int indexZ = z * imageSize.x * imageSize.y;
					for(int y = targetIndex_start_y; y < targetIndex_end_y; ++y){
						if(-1<y && y<imageSize.y){
							int indexXYZ = indexZ + y * imageSize.x + targetIndex_start_x;
							for(int x = targetIndex_start_x; x < targetIndex_end_x; ++x){
								if(-1<x && x<imageSize.x){
									const float tempTargetValue = tex1Dfetch(targetImageArray_texture, indexXYZ);
									targetValues[index + offset] = tempTargetValue;									
								}
                                else {
                                    targetValues[index + offset] = 0.0f;
                                }
								indexXYZ++;
								index++;
							}
						}
						else index+= BLOCK_WIDTH;
					}
				}
				else index+= BLOCK_WIDTH* BLOCK_WIDTH;
			}
            float3 targetPosition;
		    targetPosition.x = i * BLOCK_WIDTH;
		    targetPosition.y = j * BLOCK_WIDTH;
		    targetPosition.z = k * BLOCK_WIDTH;
		    apply_affine(targetPosition, &(targetPosition_d[currentBlockIndex * 3]));        
        }
    }
}

// CUDA kernel to process the result blocks
__global__ void process_result_blocks_gpu(float *targetPosition_d,
                                          float *resultPosition_d,
                                          float *targetValues,
                                          float *resultValues)
{

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int3 bDim = c_BlockDim;
	if (tid < bDim.x * bDim.y * bDim.z){		
        const int activeBlockIndex = tex1Dfetch(activeBlock_texture,tid);
		if (activeBlockIndex >= 0){
            int tempIndex = tid;
		    const int k =(int)(tempIndex/(bDim.x * bDim.y));
		    tempIndex -= k * bDim.x * bDim.y;
		    const int j =(int)(tempIndex/(bDim.x));
		    const int i = tempIndex - j * (bDim.x);		
            const int offset = activeBlockIndex * BLOCK_SIZE;
			const int targetIndex_start_x = i * BLOCK_WIDTH;
			const int targetIndex_start_y = j * BLOCK_WIDTH;
			const int targetIndex_start_z = k * BLOCK_WIDTH;			
            unsigned int index = 0;
            const int3 imageSize = c_ImageSize;

            __shared__ int n[Block_result_block];
            __shared__ int m[Block_result_block];
            __shared__ int l[Block_result_block];
            
            float bestCC = 0.0f;            
            float3 bestDisplacement = make_float3(0.0f, 0.0f, 0.0f);

            for(n[threadIdx.x] = -OVERLAP_SIZE; n[threadIdx.x] < OVERLAP_SIZE; n[threadIdx.x]+=STEP_SIZE){
				const int resultIndex_start_z = targetIndex_start_z + n[threadIdx.x];
				const int resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;
				
				for(m[threadIdx.x] = -OVERLAP_SIZE; m[threadIdx.x] < OVERLAP_SIZE; m[threadIdx.x]+=STEP_SIZE){
					const int resultIndex_start_y = targetIndex_start_y + m[threadIdx.x];
					const int resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;

					for(l[threadIdx.x] = -OVERLAP_SIZE; l[threadIdx.x] < OVERLAP_SIZE; l[threadIdx.x]+=STEP_SIZE){                        
						const int resultIndex_start_x = targetIndex_start_x + l[threadIdx.x];
						const int resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
						index = 0;					
						for(int z = resultIndex_start_z; z < resultIndex_end_z; ++z){
							if(-1<z && z <imageSize.z)
                            {
								int indexZ = z * imageSize.y * imageSize.x;
								for(int y = resultIndex_start_y; y < resultIndex_end_y; ++y){
									if(-1<y && y < imageSize.y)
                                    {
										int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
										for(int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
											if(-1<x && x<imageSize.x){
												const float tempResultValue = tex1Dfetch(resultImageArray_texture, indexXYZ);
												resultValues[offset+index] = tempResultValue;
											}
                                            else 
                                            {
                                                resultValues[offset+index] = 0.0f;
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
                        
                        // Do the cross corelation stuff
                        float targetMean=0.0f;
						float resultMean=0.0f;
						float voxelNumber=0.0f;

						for(int a = 0; a < BLOCK_SIZE; ++a){
                            float tv = targetValues[a + offset];
                            float rv = resultValues[a + offset];
                            if (tv > 0.0f && rv > 0.0f) {							
								targetMean += tv;
								resultMean += rv;
								voxelNumber++;
							}
                        }

                        float targetVar=0.0f;
						float resultVar=0.0f;
                        float localCC = 0.0f;                        
						float targetTemp=0.0f;
						float resultTemp=0.0f;
						
						if(voxelNumber > 0.0f){
							targetMean /= voxelNumber;
							resultMean /= voxelNumber;
	
							for(int a = 0; a < BLOCK_SIZE; ++a)
							{
                                float tv = targetValues[a + offset];
                                float rv = resultValues[a + offset];								
                                if (tv > 0.0f && rv > 0.0f)
								{
									targetTemp = tv-targetMean;
									resultTemp = rv-resultMean;                                    
									targetVar += (targetTemp)*(targetTemp);
									resultVar += (resultTemp)*(resultTemp);                                    
									localCC += (targetTemp)*(resultTemp);
								}
							}
							
							targetVar = sqrtf(targetVar/voxelNumber);
							resultVar = sqrtf(resultVar/voxelNumber);
	
							localCC = fabsf(localCC/
								(voxelNumber*targetVar*resultVar));

                            if (localCC > bestCC) {								
                                bestCC = localCC;
                                bestDisplacement.x=l[threadIdx.x];
								bestDisplacement.y=m[threadIdx.x];
								bestDisplacement.z=n[threadIdx.x];
							}
                        }						
                    }
                }
            }
            float3 resultPosition;
			resultPosition.x = i * BLOCK_WIDTH;
			resultPosition.y = j * BLOCK_WIDTH;
			resultPosition.z = k * BLOCK_WIDTH;                        			
			bestDisplacement.x += resultPosition.x; 
            bestDisplacement.y += resultPosition.y;  
            bestDisplacement.z += resultPosition.z;
            apply_affine(bestDisplacement, &(resultPosition_d[activeBlockIndex * 3]));

        }
    }
}


#endif

