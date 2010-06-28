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

void reg_getVoxelBasedNMIGradientUsingPW_gpu(   nifti_image *targetImage,
                                                nifti_image *resultImage,
                                                float **targetImageArray_d,
                                                float **resultImageArray_d,
                                                float4 **resultGradientArray_d,
                                                float **logJointHistogram_d,
                                                float4 **voxelNMIGradientArray_d,
                                                int **mask_d,
                                                int activeVoxelNumber,
                                                double *entropies,
                                                int binning)
{
	const int voxelNumber = targetImage->nvox;
	const int binNumber = binning*(binning+2);
    const float4 entropies_h=make_float4((float)entropies[0],(float)entropies[1],(float)entropies[2],(float)entropies[3]);
	const float NMI = (entropies[0]+entropies[1])/entropies[2];

	// Bind Symbols
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Binning,&binning,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Entropies,&entropies_h,sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NMI,&NMI,sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)));

	// Texture binding
	CUDA_SAFE_CALL(cudaBindTexture(0, targetImageTexture, *targetImageArray_d, voxelNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, resultImageTexture, *resultImageArray_d, voxelNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, resultImageGradientTexture, *resultGradientArray_d, voxelNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)));

    CUDA_SAFE_CALL(cudaMemset(*voxelNMIGradientArray_d, 0, voxelNumber*sizeof(float4)));

	const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW = 
		(unsigned int)ceil((float)activeVoxelNumber/(float)Block_reg_getVoxelBasedNMIGradientUsingPW);
	dim3 B1(Block_reg_getVoxelBasedNMIGradientUsingPW,1,1);
	dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW,1,1);

	reg_getVoxelBasedNMIGradientUsingPW_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[DEBUG] reg_getVoxelBasedNMIGradientUsingPW_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

#endif
