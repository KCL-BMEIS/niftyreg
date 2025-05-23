/*
 * @file _reg_ssd_gpu.h
 * @author Marc Modat
 * @date 14/11/2012
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "CudaTools.hpp"
#include "_reg_measure_gpu.h"
#include "_reg_ssd.h"

/* *************************************************************** */
/// @brief SSD measure of similarity class on the device
class reg_ssd_gpu: public reg_ssd, public reg_measure_gpu {
public:
    /// @brief reg_ssd class constructor
    reg_ssd_gpu();
    /// @brief Measure class destructor
    virtual ~reg_ssd_gpu();

    /// @brief Initialise the reg_ssd object
    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
                                   int *refMask,
                                   int *refMaskCuda,
                                   size_t activeVoxNum,
                                   nifti_image *warpedImg,
                                   float *warpedImgCuda,
                                   nifti_image *warpedGrad,
                                   float4 *warpedGradCuda,
                                   nifti_image *voxelBasedGrad,
                                   float4 *voxelBasedGradCuda,
                                   nifti_image *localWeightSim = nullptr,
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) override;
    /// @brief Returns the ssd value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the ssd value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based ssd gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based ssd gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;
};
/* *************************************************************** */
