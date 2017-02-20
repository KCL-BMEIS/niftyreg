/*
 *  _reg_kld.h
 *
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_KLDIV_H
#define _REG_KLDIV_H

#include "_reg_measure.h"

/* *************************************************************** */
class reg_kld : public reg_measure
{
public:
   /// @brief reg_kld class constructor
   reg_kld();
   /// @brief Initialise the reg_kld object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          nifti_image *forwardLocalWeightPtr = NULL,
                          int *maskFloPtr = NULL,
                          nifti_image *warRefImgPtr = NULL,
                          nifti_image *warRefGraPtr = NULL,
                          nifti_image *bckVoxBasedGraPtr = NULL);
   /// @brief Returns the kld value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based kld gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint);
   /// @brief reg_kld class destructor
   ~reg_kld() {}
};
/* *************************************************************** */

/** @brief Computes and returns the KLD between two input image
 * @param reference First input image to use to compute the metric
 * @param warped Second input image to use to compute the metric
 * @param activeTimePoint Specified which time point volumes have to be considered
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the KLD. The argument is ignored if the
 * pointer is set to NULL
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns the computed sum squared difference
 */
extern "C++" template <class DTYPE>
double reg_getKLDivergence(nifti_image *reference,
                           nifti_image *warped,
                           double *timePointWeight,
                           nifti_image *jacobianDeterminantImage,
                           int *mask);
/* *************************************************************** */

/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param reference First input image to use to compute the metric
 * @param warped Second input image to use to compute the metric
 * @param activeTimePoint Specified which time point volumes have to be considered
 * @param warpedGradient Spatial gradient of the input result image
 * @param KLdivGradient Output image htat will be updated with the
 * value of the KLD gradient
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the KLD. The argument is ignored if the
 * pointer is set to NULL
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 */
extern "C++" template <class DTYPE>
void reg_getKLDivergenceVoxelBasedGradient(nifti_image *reference,
                                           nifti_image *warped,
                                           nifti_image *warpedGradient,
                                           nifti_image *KLdivGradient,
                                           nifti_image *jacobianDeterminantImage,
                                           int *mask,
                                           int current_timepoint,
                                 double timepoint_weight);
/* *************************************************************** */

#endif
