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
class reg_kld: public reg_measure {
public:
    /// @brief reg_kld class constructor
    reg_kld();
    /// @brief reg_kld class destructor
    virtual ~reg_kld() {}

    /// @brief Initialise the reg_kld object
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
    /// @brief Returns the kld value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the kld value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based kld gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based kld gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;
};
/* *************************************************************** */
