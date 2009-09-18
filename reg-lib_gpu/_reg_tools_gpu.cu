/*
 *  _reg_tools_gpu.cu
 *  
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_GPU_CU
#define _REG_TOOLS_GPU_CU

#include "_reg_blocksize_gpu.h"
#include "_reg_tools_kernels.cu"


void reg_voxelCentric2NodeCentric_gpu(	nifti_image *targetImage,
					nifti_image *controlPointImage,
					float4 **voxelNMIGradientArray_d,
					float4 **nodeNMIGradientArray_d)
{
	const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
	const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
	const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
	const int3 gridSize = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
	const float3 voxelNodeRatio_h = make_float3(
		controlPointImage->dx / targetImage->dx,
		controlPointImage->dy / targetImage->dy,
		controlPointImage->dz / targetImage->dz);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&gridSize,sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNodeRatio,&voxelNodeRatio_h,sizeof(float3)));

	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *voxelNMIGradientArray_d, voxelNumber*sizeof(float4)));

	const unsigned int Grid_reg_voxelCentric2NodeCentric = (unsigned int)ceil((float)nodeNumber/(float)Block_reg_voxelCentric2NodeCentric);
	dim3 B1(Block_reg_voxelCentric2NodeCentric,1,1);
	dim3 G1(Grid_reg_voxelCentric2NodeCentric,1,1);

	reg_voxelCentric2NodeCentric_kernel <<< G1, B1 >>> (*nodeNMIGradientArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_voxelCentric2NodeCentric_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

void reg_convertNMIGradientFromVoxelToRealSpace_gpu(	mat44 *sourceMatrix_xyz,
							nifti_image *controlPointImage,
							float4 **nodeNMIGradientArray_d)
{
	const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));

	float4 *matrix_h;CUDA_SAFE_CALL(cudaMallocHost((void **)&matrix_h, 3*sizeof(float4)));
	matrix_h[0] = make_float4(sourceMatrix_xyz->m[0][0], sourceMatrix_xyz->m[0][1], sourceMatrix_xyz->m[0][2], sourceMatrix_xyz->m[0][3]);
	matrix_h[1] = make_float4(sourceMatrix_xyz->m[1][0], sourceMatrix_xyz->m[1][1], sourceMatrix_xyz->m[1][2], sourceMatrix_xyz->m[1][3]);
	matrix_h[2] = make_float4(sourceMatrix_xyz->m[2][0], sourceMatrix_xyz->m[2][1], sourceMatrix_xyz->m[2][2], sourceMatrix_xyz->m[2][3]);
	float4 *matrix_d;
	CUDA_SAFE_CALL(cudaMalloc((void **)&matrix_d, 3*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpy(matrix_d, matrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost((void *)matrix_h));
	CUDA_SAFE_CALL(cudaBindTexture(0, matrixTexture, matrix_d, 3*sizeof(float4)));
	
	const unsigned int Grid_reg_convertNMIGradientFromVoxelToRealSpace =
		(unsigned int)ceil((float)nodeNumber/(float)Block_reg_convertNMIGradientFromVoxelToRealSpace);
	dim3 B1(Grid_reg_convertNMIGradientFromVoxelToRealSpace,1,1);
	dim3 G1(Block_reg_convertNMIGradientFromVoxelToRealSpace,1,1);

	_reg_convertNMIGradientFromVoxelToRealSpace_kernel <<< G1, B1 >>> (*nodeNMIGradientArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_convertNMIGradientFromVoxelToRealSpace: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
	CUDA_SAFE_CALL(cudaFree(matrix_d));
}


void reg_initialiseConjugateGradient(	float4 **nodeNMIGradientArray_d,
					float4 **conjugateG_d,
					float4 **conjugateH_d,
					int nodeNumber)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)));

	const unsigned int Grid_reg_initialiseConjugateGradient =
		(unsigned int)ceil((float)nodeNumber/(float)Block_reg_initialiseConjugateGradient);
	dim3 B1(Grid_reg_initialiseConjugateGradient,1,1);
	dim3 G1(Block_reg_initialiseConjugateGradient,1,1);

	reg_initialiseConjugateGradient_kernel <<< G1, B1 >>> (*conjugateG_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_initialiseConjugateGradient: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
	CUDA_SAFE_CALL(cudaMemcpy(*conjugateH_d, *conjugateG_d, nodeNumber*sizeof(float4), cudaMemcpyDeviceToDevice));
}

void reg_GetConjugateGradient(	float4 **nodeNMIGradientArray_d,
				float4 **conjugateG_d,
				float4 **conjugateH_d,
				int nodeNumber)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, conjugateGTexture, *conjugateG_d, nodeNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, conjugateHTexture, *conjugateH_d, nodeNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)));

	// gam = sum((grad+g)*grad)/sum(HxG);
	const unsigned int Grid_reg_GetConjugateGradient1 = (unsigned int)ceil((float)nodeNumber/(float)Block_reg_GetConjugateGradient1);
	dim3 B1(Block_reg_GetConjugateGradient1,1,1);
	dim3 G1(Grid_reg_GetConjugateGradient1,1,1);

	float2 *sum_d;
	CUDA_SAFE_CALL(cudaMalloc((void **)&sum_d, nodeNumber*sizeof(float2)));
	reg_GetConjugateGradient1_kernel <<< G1, B1 >>> (sum_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_GetConjugateGradient1 kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
	float2 *sum_h;CUDA_SAFE_CALL(cudaMallocHost((void **)&sum_h, nodeNumber*sizeof(float2)));
	CUDA_SAFE_CALL(cudaMemcpy(sum_h,sum_d, nodeNumber*sizeof(float2),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(sum_d));
	double dgg = 0.0;
	double gg = 0.0;
	for(int i=0; i<nodeNumber; i++){
		dgg += sum_h[i].x;
		gg += sum_h[i].y;
	}
	float gam = dgg / gg;
	CUDA_SAFE_CALL(cudaFreeHost((void *)sum_h));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor,&gam,sizeof(float)));
	const unsigned int Grid_reg_GetConjugateGradient2 = (unsigned int)ceil((float)nodeNumber/(float)Block_reg_GetConjugateGradient2);
	dim3 B2(Block_reg_GetConjugateGradient2,1,1);
	dim3 G2(Grid_reg_GetConjugateGradient2,1,1);
	reg_GetConjugateGradient2_kernel <<< G2, B2 >>> (*nodeNMIGradientArray_d, *conjugateG_d, *conjugateH_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_GetConjugateGradient2 kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif


}

float reg_getMaximalLength_gpu(	float4 **nodeNMIGradientArray_d,
				int nodeNumber)
{

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)));

	// each thread extract the maximal value out of 128
	const int threadNumber = (int)ceil((float)nodeNumber/128.0f);
	const unsigned int Grid_reg_getMaximalLength = (unsigned int)ceil((float)threadNumber/(float)Block_reg_getMaximalLength);
	dim3 B1(Block_reg_getMaximalLength,1,1);
	dim3 G1(Grid_reg_getMaximalLength,1,1);

	float *all_d;
	CUDA_SAFE_CALL(cudaMalloc((void **)&all_d, threadNumber*sizeof(float)));
	reg_getMaximalLength_kernel <<< G1, B1 >>> (all_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_getMaximalLength kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
	float *all_h;CUDA_SAFE_CALL(cudaMallocHost((void **)&all_h, nodeNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(all_h, all_d, threadNumber*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(all_d));
	double maxDistance = 0.0f;
	for(int i=0; i<threadNumber; i++) maxDistance = all_h[i]>maxDistance?all_h[i]:maxDistance;
	CUDA_SAFE_CALL(cudaFreeHost((void *)all_h));

	return maxDistance;
}

void reg_updateControlPointPosition_gpu(nifti_image *controlPointImage,
					float4 **controlPointImageArray_d,
					float4 **bestControlPointPosition_d,
					float4 **nodeNMIGradientArray_d,
					float currentLength)
{
	const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor,&currentLength,sizeof(float)));

	CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *bestControlPointPosition_d, nodeNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)));

	const unsigned int Grid_reg_updateControlPointPosition = (unsigned int)ceil((float)nodeNumber/(float)Block_reg_updateControlPointPosition);
	dim3 B1(Block_reg_updateControlPointPosition,1,1);
	dim3 G1(Grid_reg_updateControlPointPosition,1,1);

	reg_updateControlPointPosition_kernel <<< G1, B1 >>> (*controlPointImageArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_updateControlPointPosition kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

#endif

