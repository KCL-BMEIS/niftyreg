/*
 *  _reg_mutualinformation_gpu.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_GPU_CU
#define _REG_MUTUALINFORMATION_GPU_CU

#include "_reg_blocksize_gpu.h"
#include "_reg_mutualinformation_gpu.h"
#include "_reg_mutualinformation_kernels.cu"

void reg_getVoxelBasedNMIGradientUsingPW_gpu(	nifti_image *targetImage,
						nifti_image *resultImage,
						float **targetImageArray_d,
						float **resultImageArray_d,
						float4 **resultGradientArray_d,
						float **logJointHistogram_d,
						float4 **voxelNMIGradientArray_d,
						double *entropies,
						int binning,
						bool includePadding)
{
	const int voxelNumber = targetImage->nvox;
	const int binNumber = binning*(binning+2);
	const float4 entropies_h=make_float4(entropies[0],entropies[1],entropies[2],entropies[3]);
	const float NMI = (entropies[0]+entropies[1])/entropies[2];

	// Bind Symbols
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Binning,&binning,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Entropies,&entropies_h,sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NMI,&NMI,sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_IncludePadding,&includePadding,sizeof(bool)));

	// Texture binding
	CUDA_SAFE_CALL(cudaBindTexture(0, targetImageTexture, *targetImageArray_d, voxelNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, resultImageTexture, *resultImageArray_d, voxelNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, resultImageGradientTexture, *resultGradientArray_d, voxelNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
	
	const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW = 
		(unsigned int)ceil((float)voxelNumber/(float)Block_reg_getVoxelBasedNMIGradientUsingPW);
	dim3 B1(Block_reg_getVoxelBasedNMIGradientUsingPW,1,1);
	dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW,1,1);

	reg_getVoxelBasedNMIGradientUsingPW_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_getVoxelBasedNMIGradientUsingPW_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

void reg_smoothImageForCubicSpline_gpu(	nifti_image *resultImage,
					float4 **voxelNMIGradientArray_d,
					int *smoothingRadius)
{
	const int voxelNumber = resultImage->nvox;
	int windowSize;
	float4 *smoothedImage;
	float *window;
	const int3 imageSize = make_int3(resultImage->nx, resultImage->ny, resultImage->nz);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize,&imageSize,sizeof(int3)));

	// X axis 
	windowSize = 1+smoothingRadius[0]*2;
	CUDA_SAFE_CALL(cudaMalloc((void **)&window,windowSize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&smoothedImage,voxelNumber*sizeof(float4)));
	int Grid_reg_FillConvolutionWindows = (int)ceil((float)windowSize/(float)Block_reg_FillConvolutionWindows);
	dim3 B1(Block_reg_FillConvolutionWindows,1,1);
	dim3 G1(Grid_reg_FillConvolutionWindows,1,1);
	FillConvolutionWindows_kernel <<< G1, B1 >>> (window, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] FillConvolutionWindows_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
	const unsigned int Grid_reg_ApplyConvolutionWindowAlongX =
		(unsigned int)ceil((float)voxelNumber/(float)Block_reg_ApplyConvolutionWindowAlongX);
	dim3 B2(Block_reg_ApplyConvolutionWindowAlongX,1,1);
	dim3 G2(Grid_reg_ApplyConvolutionWindowAlongX,1,1);
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *voxelNMIGradientArray_d, voxelNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, convolutionWinTexture, window, windowSize*sizeof(float)));
	_reg_ApplyConvolutionWindowAlongX_kernel <<< G2, B2 >>> (smoothedImage, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_ApplyConvolutionWindowAlongX_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
	CUDA_SAFE_CALL(cudaFree(window));
	CUDA_SAFE_CALL(cudaMemcpy(*voxelNMIGradientArray_d, smoothedImage, voxelNumber*sizeof(float4), cudaMemcpyDeviceToDevice));

	// Y axis 
	windowSize = 1+smoothingRadius[1]*2;
	CUDA_SAFE_CALL(cudaMalloc((void **)&window,windowSize*sizeof(float)));
	Grid_reg_FillConvolutionWindows = (int)ceil((float)windowSize/(float)Block_reg_FillConvolutionWindows);
	dim3 B3(Block_reg_FillConvolutionWindows,1,1);
	dim3 G3(Grid_reg_FillConvolutionWindows,1,1);
	FillConvolutionWindows_kernel <<< G3, B3 >>> (window, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] FillConvolutionWindows_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G3.x,G3.y,G3.z,B3.x,B3.y,B3.z);
#endif
	const unsigned int Grid_reg_ApplyConvolutionWindowAlongY =
		(unsigned int)ceil((float)voxelNumber/(float)Block_reg_ApplyConvolutionWindowAlongY);
	dim3 B4(Block_reg_ApplyConvolutionWindowAlongY,1,1);
	dim3 G4(Grid_reg_ApplyConvolutionWindowAlongY,1,1);
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *voxelNMIGradientArray_d, voxelNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, convolutionWinTexture, window, windowSize*sizeof(float)));
	_reg_ApplyConvolutionWindowAlongY_kernel <<< G4, B4 >>> (smoothedImage, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_ApplyConvolutionWindowAlongY_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G4.x,G4.y,G4.z,B4.x,B4.y,B4.z);
#endif
	CUDA_SAFE_CALL(cudaFree(window));
	CUDA_SAFE_CALL(cudaMemcpy(*voxelNMIGradientArray_d, smoothedImage, voxelNumber*sizeof(float4), cudaMemcpyDeviceToDevice));

	// Z axis 
	windowSize = 1+smoothingRadius[2]*2;
	CUDA_SAFE_CALL(cudaMalloc((void **)&window,windowSize*sizeof(float)));
	Grid_reg_FillConvolutionWindows = (int)ceil((float)windowSize/(float)Block_reg_FillConvolutionWindows);
	dim3 B5(Block_reg_FillConvolutionWindows,1,1);
	dim3 G5(Grid_reg_FillConvolutionWindows,1,1);
	FillConvolutionWindows_kernel <<< G5, B5 >>> (window, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] FillConvolutionWindows_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G5.x,G5.y,G5.z,B5.x,B5.y,B5.z);
#endif
	const unsigned int Grid_reg_ApplyConvolutionWindowAlongZ =
		(unsigned int)ceil((float)voxelNumber/(float)Block_reg_ApplyConvolutionWindowAlongZ);
	dim3 B6(Block_reg_ApplyConvolutionWindowAlongZ,1,1);
	dim3 G6(Grid_reg_ApplyConvolutionWindowAlongZ,1,1);
	CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *voxelNMIGradientArray_d, voxelNumber*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, convolutionWinTexture, window, windowSize*sizeof(float)));
	_reg_ApplyConvolutionWindowAlongZ_kernel <<< G6, B6 >>> (smoothedImage, windowSize);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if _DEBUG
	printf("[DEBUG] reg_ApplyConvolutionWindowAlongZ_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G6.x,G6.y,G6.z,B6.x,B6.y,B6.z);
#endif
	CUDA_SAFE_CALL(cudaFree(window));
	CUDA_SAFE_CALL(cudaMemcpy(*voxelNMIGradientArray_d, smoothedImage, voxelNumber*sizeof(float4), cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaFree(smoothedImage));
}
#endif
