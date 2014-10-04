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
#include <fstream>

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
//#include <thrust\copy.h>
//#include <thrust\device_ptr.h>

bool compare1( float &a) {
	return (a > -1);
}

void block_matching_method_gpu(nifti_image *targetImage, nifti_image *resultImage, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d) {
	// Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
	NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	//const unsigned int nBlocks = params->blockNumber[0]* params->blockNumber[1]* params->blockNumber[2];

	// Copy some required parameters over to the device
	int3 bDim = make_int3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_BlockDim, &bDim, sizeof(int3)));

	// Image size
	uint3 image_size = make_uint3(targetImage->nx, targetImage->ny, targetImage->nz);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize, &image_size, sizeof(uint3)));

	// Texture binding
	const unsigned int numBlocks = bDim.x * bDim.y * bDim.z;
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, activeBlock_texture, *activeBlock_d, numBlocks * sizeof(int)));

	// Copy the sform transformation matrix onto the device memort
	mat44 *xyz_mat;
	if (targetImage->sform_code > 0)
		xyz_mat = &(targetImage->sto_xyz);
	else
		xyz_mat = &(targetImage->qto_xyz);
	float4 t_m_a_h = make_float4(xyz_mat->m[0][0], xyz_mat->m[0][1], xyz_mat->m[0][2], xyz_mat->m[0][3]);
	float4 t_m_b_h = make_float4(xyz_mat->m[1][0], xyz_mat->m[1][1], xyz_mat->m[1][2], xyz_mat->m[1][3]);
	float4 t_m_c_h = make_float4(xyz_mat->m[2][0], xyz_mat->m[2][1], xyz_mat->m[2][2], xyz_mat->m[2][3]);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_a, &t_m_a_h, sizeof(float4)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_b, &t_m_b_h, sizeof(float4)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_c, &t_m_c_h, sizeof(float4)));
	// We need to allocate some memory to keep track of overlap areas and values for blocks
	size_t memSize = BLOCK_SIZE * numBlocks * sizeof(float);
	/*float * targetValues;
	NR_CUDA_SAFE_CALL(cudaMalloc(&targetValues, memSize));*/
	//memSize = BLOCK_SIZE * params->activeBlockNumber;
	float * resultValues;
	NR_CUDA_SAFE_CALL(cudaMalloc(&resultValues, memSize));
	unsigned int Grid_block_matching = (unsigned int)ceil((float)params->activeBlockNumber / (float)NR_BLOCK->Block_target_block);
	unsigned int Grid_block_matching_2 = 1;

	// We have hit the limit in one dimension
	if (Grid_block_matching > 65335) {
		Grid_block_matching_2 = (unsigned int)ceil((float)Grid_block_matching / 65535.0f);
		Grid_block_matching = 65335;
	}

	mat44 targetMatrix_xyz = (targetImage->sform_code > 0) ? (targetImage->sto_xyz) : (targetImage->qto_xyz);
	float* targetMat = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(targetMatrix_xyz, targetMat);

	float* targetMat_d;
	int*activeBlockNumber_d;
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&targetMat_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(targetMat_d, targetMat, 16 * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&activeBlockNumber_d), numBlocks * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(activeBlockNumber_d, params->activeBlock, numBlocks * sizeof(int), cudaMemcpyHostToDevice));

	const unsigned int threads3d[3] = { 4, 4, 4 };

	const unsigned int bx = ceil((float)params->blockNumber[0] / (float)threads3d[0]);
	const unsigned int by = ceil((float)params->blockNumber[1] / (float)threads3d[1]);
	const unsigned int bz = ceil((float)params->blockNumber[2] / (float)threads3d[2]);


	const unsigned int b1x = ceil((float)params->blockNumber[0] / (float)8);
	const unsigned int b1y = ceil((float)params->blockNumber[1] / (float)8);
	const unsigned int b1z = ceil((float)params->blockNumber[2] / (float)8);

	const uint3 blockDims = make_uint3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	const uint3 blockSize = make_uint3(4, 4, 4);


	dim3 B1(NR_BLOCK->Block_target_block, 1, 1);
	dim3 G1(Grid_block_matching, Grid_block_matching_2, 1);
	//process the target blocks
	//process_target_blocks_gpu << <G1, B1 >> >(*targetPosition_d, targetValues);

	printf("ab: %d-%d-%d\n", params->activeBlock[0], params->activeBlock[1], params->activeBlock[2]);
	printf("bn: %d-%d-%d\n", params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	printf("defined active: %d | abs: %d\n", params->definedActiveBlock, params->activeBlockNumber);

	dim3 BlockDims3D(threads3d[0], threads3d[1], threads3d[2]);
	dim3 BlockDims1D(64, 1, 1);
	dim3 ImageGrid3D(bx, by, bz);
	dim3 BlocksGrid3D(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	targetPosKernel << <ImageGrid3D, BlockDims3D >> >(*targetPosition_d, targetMat_d, activeBlockNumber_d, blockDims);
	NR_CUDA_CHECK_KERNEL(ImageGrid3D, BlockDims3D)


	/*targetBlocksKernel << <BlocksGrid3D, BlockDims3D >> >(*targetPosition_d, targetValues);
	NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims3D)
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());*/
	


	unsigned int Result_block_matching = params->activeBlockNumber;
	unsigned int Result_block_matching_2 = 1;

	// We have hit the limit in one dimension
	if (Result_block_matching > 65335) {
		Result_block_matching_2 = (unsigned int)ceil((float)Result_block_matching / 65535.0f);
		Result_block_matching = 65335;
	}

	/*dim3 B2(NR_BLOCK->Block_result_block, 1, 1);
	dim3 G2(Result_block_matching, Result_block_matching_2, 1);*/
	float* resultVar_d; 
	float* resultTemp_d;
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&resultVar_d), targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&resultTemp_d), targetImage->nvox * sizeof(float)));

	/*dim3 I1(512, 1, 1);
	dim3 B01((BlocksGrid3D.x % 2) + (BlocksGrid3D.x / 2), (BlocksGrid3D.y % 2) + (BlocksGrid3D.y / 2), (BlocksGrid3D.z % 2) + (BlocksGrid3D.z / 2));
	preCompute << <BlocksGrid3D, BlockDims1D >> >(resultVar_d, resultTemp_d);
	NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D)*/
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());


	/*resultsKernelp2 << <B01, I1 >> >(*resultPosition_d, *mask_d, targetMat_d, blockDims);
	NR_CUDA_CHECK_KERNEL(B01, I1)*/

	resultsKernel2pp2 << <BlocksGrid3D, BlockDims1D >> >(*resultPosition_d, *mask_d, targetMat_d, blockSize);
	NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D)

	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));
	//NR_CUDA_SAFE_CALL(cudaFree(targetValues));
	NR_CUDA_SAFE_CALL(cudaFree(resultValues));

	NR_CUDA_SAFE_CALL(cudaMemcpy(params->resultPosition, *resultPosition_d, params->activeBlockNumber * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	NR_CUDA_SAFE_CALL(cudaMemcpy(params->targetPosition, *targetPosition_d, params->activeBlockNumber * 3 * sizeof(float), cudaMemcpyDeviceToHost));


	cudaFree(targetPosition_d);
	cudaFree(resultPosition_d);
	cudaFree(activeBlockNumber_d);
	cudaFree(mask_d);
	cudaFree(targetMat_d);
	cudaFree(resultTemp_d);
	cudaFree(resultVar_d);
	cudaDeviceReset();

}

void block_matching_method_gpu2(nifti_image *targetImage,
	nifti_image *resultImage,
	_reg_blockMatchingParam *params,
	float **targetImageArray_d,
	float **resultImageArray_d,
	float **targetPosition_d,
	float **resultPosition_d,
	int **activeBlock_d)
{


	//this way, we can try different block sizes. My guess is that a dynamic size, will suit better. Larger sizes, will take care of the first big moves
	//and finer boxes, will fine tune towards convergence
	/*const unsigned int blockDimx = 4;
	const unsigned int blockDimy = 4;
	const unsigned int blockDimz = 4;

	const unsigned int xbsInt = targetImage->nx / (blockDimx * 2);
	const unsigned int ybsInt = targetImage->ny / (blockDimy * 2);
	const unsigned int zbsInt = targetImage->nz / (blockDimz * 2);

	const unsigned int nBlocksx = (targetImage->nx % blockDimx) ? xbsInt : xbsInt + 1;
	const unsigned int nBlocksy = (targetImage->ny % blockDimy) ? ybsInt : ybsInt + 1;
	const unsigned int nBlocksz = (targetImage->nz % blockDimz) ? zbsInt : zbsInt + 1;*/
	size_t      memSize2 = params->activeBlockNumber * 3 * sizeof(float);
	// Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
	NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	if (resultImage != resultImage)
		printf("Useless lines to avoid a warning");

	// Copy some required parameters over to the device
	int3 bDim = make_int3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_BlockDim, &bDim, sizeof(int3)));

	// Image size
	int3 image_size = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize, &image_size, sizeof(int3)));

	// Texture binding
	const int numBlocks = bDim.x*bDim.y*bDim.z;
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox*sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox*sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, activeBlock_texture, *activeBlock_d, numBlocks*sizeof(int)));

	// Copy the sform transformation matrix onto the device memort
	mat44 *xyz_mat;
	if (targetImage->sform_code > 0)
		xyz_mat = &(targetImage->sto_xyz);
	else xyz_mat = &(targetImage->qto_xyz);
	float4 t_m_a_h = make_float4(xyz_mat->m[0][0], xyz_mat->m[0][1], xyz_mat->m[0][2], xyz_mat->m[0][3]);
	float4 t_m_b_h = make_float4(xyz_mat->m[1][0], xyz_mat->m[1][1], xyz_mat->m[1][2], xyz_mat->m[1][3]);
	float4 t_m_c_h = make_float4(xyz_mat->m[2][0], xyz_mat->m[2][1], xyz_mat->m[2][2], xyz_mat->m[2][3]);
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_a, &t_m_a_h, sizeof(float4)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_b, &t_m_b_h, sizeof(float4)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_c, &t_m_c_h, sizeof(float4)));
	// We need to allocate some memory to keep track of overlap areas and values for blocks
	unsigned int memSize = BLOCK_SIZE * params->activeBlockNumber;
	//printf("memsize: %d | abn: %d - %d\n", memSize, params->activeBlockNumber, numBlocks);


	float * values;
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)&values, BLOCK_SIZE *numBlocks * sizeof(float)));

	unsigned int Grid_block_matching = (unsigned int)ceil((float)params->activeBlockNumber / (float)NR_BLOCK->Block_target_block);
	unsigned int Grid_block_matching_2 = 1;

	// We have hit the limit in one dimension
	if (Grid_block_matching > 65335) {
		Grid_block_matching_2 = (unsigned int)ceil((float)Grid_block_matching / 65535.0f);
		Grid_block_matching = 65335;
	}

	dim3 B1(NR_BLOCK->Block_target_block, 1, 1);
	dim3 G1(Grid_block_matching, Grid_block_matching_2, 1);
	printf("blocks: %d | threads: %d \n", Grid_block_matching, NR_BLOCK->Block_target_block);
	// process the target blocks
	process_target_blocks_gpu << <G1, B1 >> >(*targetPosition_d, values);
	NR_CUDA_CHECK_KERNEL(G1, B1)
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[NiftyReg CUDA DEBUG] process_target_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
		cudaGetErrorString(cudaGetLastError()), G1.x, G1.y, G1.z, B1.x, B1.y, B1.z);
#endif





	unsigned int Result_block_matching = params->activeBlockNumber / (float)NR_BLOCK->Block_result_block;
	unsigned int Result_block_matching_2 = 1;

	// We have hit the limit in one dimension
	if (Result_block_matching > 65335) {
		Result_block_matching_2 = (unsigned int)ceil((float)Result_block_matching / 65535.0f);
		Result_block_matching = 65335;
	}

	dim3 B2(NR_BLOCK->Block_result_block, 1, 1);
	dim3 G2(Result_block_matching, Result_block_matching_2, 1);
	printf("blocks: %d | threads: %d \n", Result_block_matching, NR_BLOCK->Block_result_block);
	resultBlocksKernel << <G2, B2 >> >(*resultPosition_d, values);
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[NiftyReg CUDA DEBUG] process_result_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
		cudaGetErrorString(cudaGetLastError()), G2.x, G2.y, G2.z, B2.x, B2.y, B2.z);
#endif
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));
	NR_CUDA_SAFE_CALL(cudaFree(values));

	// We will simply call the CPU version as this step is probably
	// not worth implementing on the GPU.
	// device to host copy

	/*NR_CUDA_SAFE_CALL(cudaMemcpy(params->targetPosition, *targetPosition_d, memSize2, cudaMemcpyDeviceToHost));*/
	NR_CUDA_SAFE_CALL(cudaMemcpy(params->resultPosition, *resultPosition_d, memSize2, cudaMemcpyDeviceToHost));
	NR_CUDA_SAFE_CALL(cudaMemcpy(params->targetPosition, *targetPosition_d, memSize2, cudaMemcpyDeviceToHost));


	cudaFree(targetPosition_d);
	cudaFree(resultPosition_d);

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
