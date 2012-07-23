/**
 * @file _reg_ssd.h
 * @brief File that contains sum squared difference related function
 * @author Marc Modat
 * @date 19/05/2009
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_H
#define _REG_SSD_H

#include "nifti1_io.h"

/** @brief Copmutes and returns the SSD between two input image
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SSD. The argument is ignored if the
 * pointer is set to NULL
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns the computed sum squared difference
 */
extern "C++"
double reg_getSSD(nifti_image *targetImage,
                  nifti_image *resultImage,
                  nifti_image *jacobianDeterminantImage,
                  int *mask
                  );

/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param resultImageGradient Spatial gradient of the input result image
 * @param ssdGradientImage Output image htat will be updated with the
 * value of the SSD gradient
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SSD. The argument is ignored if the
 * pointer is set to NULL
 * @param maxSD Input scalar that contain the difference value between
 * the highest and the lowest intensity.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 */
extern "C++"
void reg_getVoxelBasedSSDGradient(nifti_image *targetImage,
                                  nifti_image *resultImage,
                                  nifti_image *resultImageGradient,
                                  nifti_image *ssdGradientImage,
                                  nifti_image *jacobianDeterminantImage,
                                  float maxSD,
                                  int *mask
                                  );
#endif
