/*
 *  _reg_blockMatching_gpu.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright 2009 UCL - CMIC. All rights reserved.
 *
 */

#ifndef _REG_BLOCKMATCHING_GPU_CU
#define _REG_BLOCKMATCHING_GPU_CU

#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching_kernels.cu"

#include "_reg_blocksize_gpu.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"




void block_matching_method_gpu3(nifti_image *targetImage, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d) {

	// Copy some required parameters over to the device
	int3 bDim = make_int3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	uint3 image_size = make_uint3(targetImage->nx, targetImage->ny, targetImage->nz);// Image size
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_BlockDim, &bDim, sizeof(uint3)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize, &image_size, sizeof(uint3)));

	// Texture binding
	const unsigned int numBlocks = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, activeBlock_texture, *activeBlock_d, numBlocks * sizeof(int)));

	mat44 targetMatrix_xyz = (targetImage->sform_code > 0) ? (targetImage->sto_xyz) : (targetImage->qto_xyz);
	float* targetMat = (float *)malloc(16 * sizeof(float));//freed
	mat44ToCptr(targetMatrix_xyz, targetMat);

	float* targetMat_d;//freed
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&targetMat_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(targetMat_d, targetMat, 16 * sizeof(float), cudaMemcpyHostToDevice));

	unsigned int* definedBlock_d;
	unsigned int *definedBlock_h = (unsigned int*)malloc(sizeof(unsigned int));
	definedBlock_h[0] = 0;
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&definedBlock_d), sizeof(unsigned int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(definedBlock_d, definedBlock_h, sizeof(unsigned int), cudaMemcpyHostToDevice));



	dim3 BlockDims1D(64, 1, 1);
	dim3 BlocksGrid3D(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	const uint3 blockSize = make_uint3(4, 4, 4);


	blockMatchingKernel << <BlocksGrid3D, BlockDims1D >> >(*resultPosition_d, *targetPosition_d, *mask_d, targetMat_d, blockSize, definedBlock_d);
	//NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D)

	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	NR_CUDA_SAFE_CALL(cudaMemcpy((void *)definedBlock_h, (void *)definedBlock_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	params->definedActiveBlock = definedBlock_h[0];
	//printf("definedActiveBlock: %d\n", params->definedActiveBlock);
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));

	free(definedBlock_h);
	free(targetMat);
	cudaFree(targetMat_d);
	cudaFree(definedBlock_d);

}


void optimize_gpu(_reg_blockMatchingParam *blockMatchingParams,
	mat44 *updateAffineMatrix,
	float **targetPosition_d,
	float **resultPosition_d,
	bool affine)
{

	// Cheat and call the CPU version.
	optimize(blockMatchingParams, updateAffineMatrix, affine);

}

#endif
