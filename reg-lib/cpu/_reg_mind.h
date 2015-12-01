/*
 *  _reg_mind.h
 *
 *
 *  Created by Marc Modat on 01/12/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef _REG_MIND_H
#define _REG_MIND_H

#include "_reg_ssd.h"

/* *************************************************************** */
/* *************************************************************** */
/// @brief SSD measure of similarity classe
class reg_mind : public reg_ssd
{
public:
   /// @brief reg_ssd class constructor
   reg_mind();
   /// @brief Initialise the reg_ssd object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          int *maskFloPtr = NULL,
                          nifti_image *warRefImgPtr = NULL,
                          nifti_image *warRefGraPtr = NULL,
                          nifti_image *bckVoxBasedGraPtr = NULL);
   /// @brief Returns the ssd value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based ssd gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Measure class desstructor
   ~reg_mind();
protected:
   nifti_image *referenceImageDescriptor;
   nifti_image *warpedImageDescriptor;
   /// @brief Compute the MIND descriptor of the reference image
   void GetReferenceImageDesciptor();
   /// @brief Compute the MIND descriptor of the warped image
   void GetWarpedImageDesciptor();
};
#endif
