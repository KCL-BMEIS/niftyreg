/**
 * @file  _reg_lncc.h
 * @author Aileen Corder
 * @author Marc Modat
 * @date 10/11/2012.
 * @brief Header file for the LNCC related class and functions
 * Copyright (c) 2012-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

#include "_reg_measure.h"

/* *************************************************************** */
class reg_lncc: public reg_measure {
public:
    /// @brief reg_lncc class constructor
    reg_lncc();
    /// @brief reg_lncc class destructor
    virtual ~reg_lncc();

    /// @brief Initialise the reg_lncc object
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
    /// @brief Returns the lncc value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the lncc value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based lncc gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based lncc gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;
    /// @brief Set the kernel standard deviation
    virtual void SetKernelStandardDeviation(int t, float stddev) {
        this->kernelStandardDeviation[t] = stddev;
    }
    /// @brief Set the kernel type
    virtual void SetKernelType(ConvKernelType t) {
        this->kernelType = t;
    }

protected:
    float kernelStandardDeviation[255];
    nifti_image *correlationImage;
    nifti_image *meanImage;
    nifti_image *sdevImage;
    nifti_image *warpedMeanImage;
    nifti_image *warpedSdevImage;
    int *forwardMask;

    nifti_image *correlationImageBw;
    nifti_image *meanImageBw;
    nifti_image *sdevImageBw;
    nifti_image *warpedMeanImageBw;
    nifti_image *warpedSdevImageBw;
    int *backwardMask;

    ConvKernelType kernelType;
};
/* *************************************************************** */
