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
#include "_reg_ReadWriteImage.h"

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
                          int *maskFloPtr = NULL,
                          nifti_image *warRefImgPtr = NULL,
                          nifti_image *warRefGraPtr = NULL,
                          nifti_image *bckVoxBasedGraPtr = NULL);
   /// @brief Returns the mind value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based mind gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Measure class desstructor
   ~reg_mind();
   //
protected:
   nifti_image *referenceImageDescriptor;
   nifti_image *floatingImageDescriptor;
   nifti_image *warpedReferenceImageDescriptor;
   nifti_image *warpedFloatingImageDescriptor;
   bool activeTimePointDescriptor[255];

private:
   //Platform* platform;
   //Kernel *convolutionKernel;
};
//
extern "C++"
void GetMINDImageDesciptor(nifti_image* inputImgPtr, nifti_image* MINDImgPtr);
extern "C++" template <class InputTYPE>
void GetMINDImageDesciptor1(nifti_image* inputImgPtr, nifti_image* MINDImgPtr);
extern "C++" template <class InputTYPE>
void ShiftImage(nifti_image* inputImgPtr, nifti_image* shiftedImgPtr, int tx, int ty, int tz);
//
#endif
