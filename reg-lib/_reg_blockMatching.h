/*
 *  _reg_blockMatching.h
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_H__
#define __REG_BLOCKMATCHING_H__

#include "nifti1_io.h"
#include <vector>
#include <iostream>

#define TOLERANCE 0.01
#define MAX_ITERATIONS 30

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define BLOCK_2D_SIZE 16
#define OVERLAP_SIZE 3
#define STEP_SIZE 1

#define NUM_BLOCKS_TO_COMPARE 343 // We compare in a 7x7x7 neighborhood.
#define NUM_BLOCKS_TO_COMPARE_2D 49
#define NUM_BLOCKS_TO_COMPARE_1D 7

/**
*
* Main algorithm of Ourselin et al.
* The essence of the algorithm is as follows:
* - Subdivide the target image into a number of blocks and find
*   the block in the result image that is most similar.
* - Get the point pair between the target and the result image block
*   for the most similar block.
*
* target: Pointer to the nifti target image.
* result: Pointer to the nifti result image.
*
*
* block_size: Size of the block.
* block_half_width: Half-width of the search neighborhood.
* delta_1: Spacing between two consecutive blocks
* delta_2: Sub-sampling value for a block
*
* Possible improvement: Take care of anisotropic data. Right now, we specify
* the block size, neighborhood and the step sizes in voxels and it would be
* better to specify it in millimeters and take the voxel size into account.
* However, it would be more efficient to calculate this once (outside this
* module) and pass these values for each axes. For the time being, we do this
* simple implementation.
*
*/

/**
 * Structure which contains the block matching parameters
 */

 #include <iostream>

struct _reg_blockMatchingParam{
        int blockNumber[3];
        int percent_to_keep;

        float * targetPosition;
        float * resultPosition;

        int activeBlockNumber;
        int *activeBlock;

        _reg_blockMatchingParam()
                :targetPosition(0), resultPosition(0), activeBlock(0)
        {}


        ~_reg_blockMatchingParam(){
                if(targetPosition) free(targetPosition);
                if(resultPosition) free(resultPosition);
                if(activeBlock) free(activeBlock);
        }
};
extern "C++"
void initialise_block_matching_method(	nifti_image * target,
                                        _reg_blockMatchingParam *params,
                                        int percentToKeep_block,
                                        int percentToKeep_opt,
                    int *mask,
                    bool runningOnGPU = false);

/**
* Interface for the block matching algorithm.
* The method actually only figures out the data type for the target image
* and the actual algoritm is executed later down the line.
*/
extern "C++"
template<typename PrecisionType>
void block_matching_method(	nifti_image * target,
                            nifti_image * result,
                            _reg_blockMatchingParam *params,
                            int *);

/**
* Apply the given affine transformation to a point
*/
void apply_affine(mat44 * mat, float *pt, float *);

/**
* Find the optimal affine transformation that matches the points
* in the target image to the point in the result image
*/
void optimize(_reg_blockMatchingParam *params, mat44 * final, bool affine = true);

/**
* A routine to perform Singular Value Decomposition on a m x n matrix.
* Inspired from Numerical recipes
*/
void svd(float ** in, int m, int n, float * w, float ** v);



#endif
