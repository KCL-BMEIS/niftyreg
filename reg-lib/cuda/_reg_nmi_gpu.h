/*
 *  _reg_mutualinformation_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_nmi.h"
#include "_reg_measure_gpu.h"

/* *************************************************************** */
/// @brief NMI measure of similarity class - GPU based
class reg_nmi_gpu: public reg_nmi, public reg_measure_gpu {
public:
    /// @brief reg_nmi_gpu class constructor
    reg_nmi_gpu();
    /// @brief reg_nmi_gpu class destructor
    virtual ~reg_nmi_gpu();

    /// @brief Initialise the reg_nmi_gpu object
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
    /// @brief Returns the nmi value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based nmi gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override;
};
/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_multichannel_nmi_gpu: public reg_multichannel_nmi, public reg_measure_gpu {
public:
    void InitialiseMeasure(nifti_image *refImg,
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
                           float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief reg_multichannel_nmi_gpu class constructor
    reg_multichannel_nmi_gpu() {}
    /// @brief reg_multichannel_nmi_gpu class destructor
    virtual ~reg_multichannel_nmi_gpu() {}
    /// @brief Returns the nmi value
    virtual double GetSimilarityMeasureValue() override { return 0; }
    /// @brief Compute the voxel based nmi gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override {}
};
/* *************************************************************** */
