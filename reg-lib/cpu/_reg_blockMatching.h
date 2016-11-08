/**
 * @file _reg_blockMatching.h
 * @brief Functions related to the block matching approach
 * @author Marc Modat and Pankaj Daga
 * @date 24/03/2009
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_H__
#define __REG_BLOCKMATCHING_H__

#include "_reg_maths.h"
#include <vector>

#define TOLERANCE 0.001
#define MAX_ITERATIONS 30

#define BLOCK_WIDTH 4
#define BLOCK_3D_SIZE 64
#define BLOCK_2D_SIZE 16
#define OVERLAP_SIZE 3

#define NUM_BLOCKS_TO_COMPARE 343 // We compare in a 7x7x7 neighborhood.
#define NUM_BLOCKS_TO_COMPARE_2D 49
#define NUM_BLOCKS_TO_COMPARE_1D 7

/// @brief Structure which contains the block matching parameters
struct _reg_blockMatchingParam
{
   int totalBlockNumber;
   int *totalBlock;
   unsigned int blockNumber[3];
   //Number of block we keep for LTS
   int percent_to_keep;

   unsigned int dim;
   float *referencePosition;
   float *warpedPosition;

   //Before:
   //Min between Number of block we keep in total (totalBlockNumber*percent_to_keep) and Number of total block - unuseable blocks
   //Now:
   //Number of total block - unuseable blocks
   int activeBlockNumber;
   //int *activeBlock;

   //Number of active block which has a displacement vector (not NaN)
   int definedActiveBlockNumber;
   //int *definedActiveBlock;

   int voxelCaptureRange;

   int stepSize;

   _reg_blockMatchingParam()
       : totalBlockNumber(0),
        totalBlock(0),
        percent_to_keep(0),
        dim(0),
        referencePosition(0),
        warpedPosition(0),
        activeBlockNumber(0),
        voxelCaptureRange(0),
        stepSize(0)
   {}

   ~_reg_blockMatchingParam()
   {
      if (referencePosition) free(referencePosition);
      if (warpedPosition) free(warpedPosition);
      if (totalBlock) free(totalBlock);
   }
};
/* *************************************************************** */
/** @brief This function initialise a _reg_blockMatchingParam structure
 * according to the the provided arguments
 * @param referenceImage Reference image where the blocks are defined
 * @param params Block matching parameter structure that will be populated
 * @param percentToKeep_block Amount of block to block to keep for the
 * optimisation process
 * @param percentToKeep_opt Hmmmm ... I actually don't remember.
 * Need to check the source :)
 * @param stepSize_block To define
 * @param mask Array than contains a mask of the voxel form the reference
 * image to consider for the registration
 * @param runningOnGPU Has to be set to true if the registration has to be performed on the GPU
 */
extern "C++"
void initialise_block_matching_method(nifti_image * referenceImage,
                                      _reg_blockMatchingParam *params,
                                      int percentToKeep_block,
                                      int percentToKeep_opt,
                                      int stepSize_block,
                                      int *mask,
                                      bool runningOnGPU = false);

/** @brief Interface for the block matching algorithm.
 * @param referenceImage Reference image in the current registration task
 * @param warpedImage Warped floating image in the currrent registration task
 * @param params Block matching parameter structure that contains all
 * relevant information
 * @param mask Mask array where only voxel defined as active are considered
 */
extern "C++"
void block_matching_method(nifti_image * referenceImage,
                           nifti_image * warpedImage,
                           _reg_blockMatchingParam *params,
                           int *mask);

/** @brief Find the optimal affine transformation that matches the points
 * in the reference image to the point in the warped image
 * @param params Block-matching structure that contains the relevant information
 * @param transformation_matrix Initial transformation matrix that is updated
 * @param affine Returns an affine transformation (12 DoFs) if set to true;
 * returns a rigid transformation (6 DoFs) otherwise
 */
void optimize(_reg_blockMatchingParam *params,
              mat44 * transformation_matrix,
              bool affine = true);
#endif
