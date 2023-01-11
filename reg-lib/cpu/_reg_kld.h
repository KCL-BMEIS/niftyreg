/*
 *  _reg_kld.h
 *
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_measure.h"

/* *************************************************************** */
class reg_kld : public reg_measure
{
public:
   /// @brief reg_kld class constructor
   reg_kld();
   /// @brief reg_kld class destructor
   virtual ~reg_kld() {}

   /// @brief Initialise the reg_kld object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          nifti_image *forwardLocalWeightPtr = nullptr,
                          int *maskFloPtr = nullptr,
                          nifti_image *warRefImgPtr = nullptr,
                          nifti_image *warRefGraPtr = nullptr,
                          nifti_image *bckVoxBasedGraPtr = nullptr);
   /// @brief Returns the kld value
   virtual double GetSimilarityMeasureValue() override;
   /// @brief Compute the voxel based kld gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override;
};
/* *************************************************************** */

/** @brief Computes and returns the KLD between two input image
 * @param reference First input image to use to compute the metric
 * @param warped Second input image to use to compute the metric
 * @param activeTimePoint Specified which time point volumes have to be considered
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the KLD. The argument is ignored if the
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to nullptr, all voxels are considered
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
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to nullptr, all voxels are considered
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
