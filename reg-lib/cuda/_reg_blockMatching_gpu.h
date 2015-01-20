/*
 *  _reg_blockMatching_gpu.h
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BLOCKMATCHING_GPU_H
#define _REG_BLOCKMATCHING_GPU_H

#include "_reg_common_gpu.h"
#include "_reg_blockMatching.h"



// targetImage: The target/fixed/reference image.
// resultImage: The warped/deformed/result image.
// blockMatchingParam:
// targetImageArray_d: The target/fixed/reference image on the device.
// targetPosition_d: Output. The center of the blocks in the target image.
// resultPosition_d: Output. The corresponding center of the blocks in the result.
// activeBlock_d: Array specifying which blocks are active.

extern "C++"
void block_matching_method_gpu(nifti_image *targetImage, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d, float** targetMat_d);


extern "C++"
void optimize_gpu(	_reg_blockMatchingParam *blockMatchingParams,
					 mat44 *updateAffineMatrix,
					 float **targetPosition_d,
					 float **resultPosition_d,
					 bool affine = true);

extern "C++"
void affineLocalSearch3DCuda(mat44 *cpuMat, float* final_d, float *A_d, float* Sigma_d, float* U_d, float* VT_d, float* r_d, float * newResultPos_d, float* targetPos_d, float* resultPos_d, float* lengths_d, const unsigned int numBlocks, const unsigned long num_to_keep, const unsigned int m, const unsigned int n) ;


extern "C++"
void optimize_affine3D_cuda(mat44* cpuMat, float* final_d, float* A_d, float* U_d, float* Sigma_d, float* VT_d, float* r_d, float* lengths_d, float* target_d, float* result_d, float* newResult_d, unsigned int m, unsigned int n,const unsigned int numToKeep, bool ilsIn);


extern "C++"
void getAffineMat3D(float* A_d, float* Sigma_d, float* VT_d, float* U_d, float* target_d, float* result_d, float* r_d, float *transformation,const unsigned int numBlocks, unsigned int m, unsigned int n);


#endif

