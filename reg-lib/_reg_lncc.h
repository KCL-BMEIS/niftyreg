/**
 * @file  _reg_lncc.h
 *
 *
 *  Created by Aileen Cordes on 10/11/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_LNCC_H
#define _REG_LNCC_H

#include "nifti1_io.h"
#include "_reg_tools.h"

/** @brief Copmutes and returns the LNCC between two input image
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 * to use.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns the computed LNCC
 */
extern "C++"
double reg_getLNCC(nifti_image *referenceImage,
                   nifti_image *warpedImage,
                   float gaussianStandardDeviation,
                   int *mask
                   );

//double reg_getLNCC_wml(nifti_image *referenceImage,
//                   nifti_image *warpedImage,
//           nifti_image *localMeanTargetImage, //
//           nifti_image *localStdTargetImage, //
//           nifti_image *localMeanResultImage, //
//           nifti_image *localStdResultImage, //
//           nifti_image *localCorrelationImage, //
//                   float gaussianStandardDeviation,
//                   int *mask
//                   , double alpha);

/** @brief Compute a voxel based gradient of the LNCC.
 *  @param targetImage First input image to use to compute the metric
 *  @param resultImage Second input image to use to compute the metric
 *  @param resultImageGradient Spatial gradient of the input result image
 *  @param lnccGradientImage Output image that will be updated with the
 *  value of the LNCC gradient
 *  @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 *  to use.
 *  @param mask Array that contains a mask to specify which voxel
 *  should be considered. If set to NULL, all voxels are considered
 */
extern "C++"
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedImageGradient,
                                   nifti_image *lnccGradientImage,
                                   float gaussianStandardDeviation,
                                   int *mask
                                   );


//void reg_getVoxelBasedLNCCGradient_wml(nifti_image *referenceImage,
//                                  nifti_image *warpedImage,
//                   nifti_image *localMeanTargetImage, //
//                           nifti_image *localStdTargetImage, //
//                           nifti_image *localMeanWarpedImage, //
//                           nifti_image *localStdWarpedImage, //
//                           nifti_image *localCorrelationImage, //
//                                  nifti_image *warpedImageGradient,
//                                  nifti_image *lnccGradientImage,
//                  float gaussianStandardDeviation,
//                                  int *mask,
//                                  double alpha
//                                  );

extern "C++"
void reg_getLocalStd(nifti_image *image,
                     nifti_image *localMeanImage,
                     nifti_image *localStdImage,
                     float gaussianStandardDeviation,
                     int *mask
                     );

extern "C++"
void reg_getLocalMean(nifti_image *image,
                      nifti_image *localMeanImage,
                      float gaussianStandardDeviation,
                      int *mask
                      );

extern "C++"
void reg_getLocalCorrelation(nifti_image *referenceImage,
                             nifti_image *warpedImage,
                             nifti_image *localMeanReferenceImage,
                             nifti_image *localMeanWarpedImage,
                             nifti_image *localCorrelationImage,
                             float gaussianStandardDeviation,
                             int *mask
                             );

extern "C++"             
void reg_meanFilter(nifti_image *image, int radius, int *mask );

extern "C++"
void reg_sobelFilter(nifti_image *image, int axis);
#endif

