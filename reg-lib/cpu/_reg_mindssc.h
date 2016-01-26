/*
 *  _reg_mindssc.h
 *
 *
 *  Created by Marc Modat on 01/12/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef _REG_MINDSSC_H
#define _REG_MINDSSC_H

#include "_reg_ssd.h"
//#include "ConvolutionKernel.h"
//#include "Platform.h"
#include <math.h>
#include "_reg_globalTrans.h"
#include "_reg_resampling.h"

/* *************************************************************** */
/* *************************************************************** */
/// @brief MINDSSC measure of similarity class
class reg_mindssc : public reg_ssd
{
public:
   /// @brief reg_mindssc class constructor
   reg_mindssc();
   /// @brief Initialise the reg_mindssc object
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
   /// @brief Returns the mindssc value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based mindssc gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Measure class desstructor
   ~reg_mindssc();
   //
protected:
   nifti_image *referenceImageDescriptor;
   nifti_image *floatingImageDescriptor;
   nifti_image *warpedReferenceImageDescriptor;
   nifti_image *warpedFloatingImageDescriptor;
   bool activeTimePointDescriptor[255];

   // gradient
   nifti_image *warpedFloatingGradientImageDescriptor;
   nifti_image *warpedReferenceGradientImageDescriptor;

private:
   //Platform* platform;
   //Kernel *convolutionKernel;
};
//

extern "C++"
void GetMINDSSCImageDesciptor(nifti_image* inputImgPtr,
                           nifti_image* MINDSSCImgPtr,
                           int *mask);

extern "C++" template <class DTYPE>
void ShiftImage(nifti_image* inputImgPtr, nifti_image* shiftedImgPtr, int* maskPtr, int tx, int ty, int tz);

extern "C++" template <class DTYPE>
void spatialGradient(nifti_image* inputImg, nifti_image* gradImg, int *mask);

#endif // _REG_MINDSSC_H
