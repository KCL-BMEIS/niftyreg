/*
 *  _reg_mind.h
 *
 *
 *  Created by Marc Modat on 01/12/2015.
 *  Copyright (c) 2015-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

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
    /// @brief Measure class destructor
    virtual ~reg_mind();

    /// @brief Initialise the reg_mind object
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

    /// @brief Returns the mind based measure of similarity value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override;

    virtual void SetDescriptorOffset(int);
    virtual int GetDescriptorOffset();

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
    /// @brief Measure class destructor
    virtual ~reg_mindssc();
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
