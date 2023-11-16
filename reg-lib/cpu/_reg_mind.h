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
#include "_reg_globalTrans.h"
#include "_reg_resampling.h"

#define MIND_TYPE 0
#define MINDSSC_TYPE 1

/* *************************************************************** */
/// @brief MIND measure of similarity class
class reg_mind: public reg_ssd {
public:
    /// @brief reg_mind class constructor
    reg_mind();
    /// @brief Measure class destructor
    virtual ~reg_mind();

    /// @brief Initialise the reg_mind object
    virtual void InitialiseMeasure(nifti_image *refImg,
                                   nifti_image *floImg,
                                   int *refMask,
                                   nifti_image *warpedImg,
                                   nifti_image *warpedGrad,
                                   nifti_image *voxelBasedGrad,
                                   nifti_image *localWeightSim = nullptr,
                                   int *floMask = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr) override;
    /// @brief Returns the forward mind-based measure of similarity value
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the backward mind-based measure of similarity value
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based mind gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based mind gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;
    virtual void SetDescriptorOffset(int val) { this->descriptorOffset = val; }
    virtual int GetDescriptorOffset() { return this->descriptorOffset; }

protected:
    nifti_image *referenceImageDescriptor;
    nifti_image *floatingImageDescriptor;
    nifti_image *warpedReferenceImageDescriptor;
    nifti_image *warpedFloatingImageDescriptor;
    double timePointWeightsDescriptor[255]{};
    int descriptorOffset;
    int mindType;
    int descriptorNumber;
};
/* *************************************************************** */
/// @brief MIND-SSC measure of similarity class
class reg_mindssc: public reg_mind {
public:
    /// @brief reg_mind class constructor
    reg_mindssc();
    /// @brief Measure class destructor
    virtual ~reg_mindssc();
};
/* *************************************************************** */
void GetMindImageDescriptor(const nifti_image *inputImage,
                            nifti_image *mindImage,
                            const int *mask,
                            const int descriptorOffset,
                            const int currentTimePoint);
/* *************************************************************** */
void GetMindSscImageDescriptor(const nifti_image *inputImage,
                               nifti_image *mindSscImage,
                               const int *mask,
                               const int descriptorOffset,
                               const int currentTimePoint);
/* *************************************************************** */
