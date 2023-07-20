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

#include "_reg_tools_gpu.h"
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
                                   cudaArray *refImgCuda,
                                   nifti_image *floImg,
                                   cudaArray *floImgCuda,
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
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) override;
    /// @brief Returns the ssd value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based ssd gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override;
};
/* *************************************************************** */
