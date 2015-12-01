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

#include "_reg_common_cuda.h"
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


#endif

