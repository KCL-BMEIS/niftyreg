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
//#include "ConvolutionKernel.h"
//#include "Platform.h"
#include <math.h>
#include "_reg_globalTrans.h"
#include "_reg_resampling.h"
#include <algorithm>

#define MIND_TYPE 0
#define MINDSSC_TYPE 1

/* *************************************************************** */
/* *************************************************************** */
/// @brief MIND measure of similarity class
class reg_mind : public reg_ssd
{
public:
   /// @brief reg_mind class constructor
   reg_mind();
   /// @brief Initialise the reg_mind object
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
   /// @brief Returns the mind based measure of similarity value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint);
   /// @brief
   void SetDescriptorOffset(int);
   int GetDescriptorOffset();
   /// @brief Measure class desstructor
   ~reg_mind();

protected:
   nifti_image *referenceImageDescriptor;
   nifti_image *floatingImageDescriptor;
   nifti_image *warpedReferenceImageDescriptor;
   nifti_image *warpedFloatingImageDescriptor;
   double timePointWeightDescriptor[255];

   int descriptorOffset;
   int mind_type;
   int discriptor_number;
};
/* *************************************************************** */
/// @brief MIND-SSC measure of similarity class
class reg_mindssc : public reg_mind
{
public:
   /// @brief reg_mind class constructor
   reg_mindssc();
   /// @brief Measure class desstructor
   ~reg_mindssc();
};
/* *************************************************************** */

extern "C++"
void GetMINDImageDesciptor(nifti_image* inputImgPtr,
                           nifti_image* MINDImgPtr,
                           int *mask,
                           int descriptorOffset,
                           int current_timepoint);
extern "C++"
void GetMINDSSCImageDesciptor(nifti_image* inputImgPtr,
                              nifti_image* MINDSSCImgPtr,
                              int *mask,
                              int descriptorOffset,
                              int current_timepoint);
#endif
