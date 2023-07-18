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
    /// @brief reg_nmi class constructor
    reg_nmi_gpu();
    /// @brief reg_nmi class destructor
    virtual ~reg_nmi_gpu();

    /// @brief Initialise the reg_nmi_gpu object
    virtual void InitialiseMeasure(nifti_image *refImg,
                                   nifti_image *floImg,
                                   int *refMask,
                                   size_t activeVoxNum,
                                   nifti_image *warpedImg,
                                   nifti_image *warpedGrad,
                                   nifti_image *voxelBasedGrad,
                                   nifti_image *localWeightSim,
                                   cudaArray *refImgCuda,
                                   cudaArray *floImgCuda,
                                   int *refMaskCuda,
                                   float *warpedImgCuda,
                                   float4 *warpedGradCuda,
                                   float4 *voxelBasedGradCuda) override;
    /// @brief Returns the nmi value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based nmi gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override;

protected:
    float *forwardJointHistogramLog_device;
    // float **backwardJointHistogramLog_device;
    void DeallocateHistogram();
};
/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_multichannel_nmi_gpu: public reg_multichannel_nmi, public reg_measure_gpu {
public:
    void InitialiseMeasure(nifti_image *refImg,
                           nifti_image *floImg,
                           int *refMask,
                           size_t activeVoxNum,
                           nifti_image *warpedImg,
                           nifti_image *warpedGrad,
                           nifti_image *voxelBasedGrad,
                           nifti_image *localWeightSim,
                           cudaArray *refImgCuda,
                           cudaArray *floImgCuda,
                           int *refMaskCuda,
                           float *warpedImgCuda,
                           float4 *warpedGradCuda,
                           float4 *voxelBasedGradCuda) override {}
    /// @brief reg_nmi class constructor
    reg_multichannel_nmi_gpu() {}
    /// @brief reg_nmi class destructor
    virtual ~reg_multichannel_nmi_gpu() {}
    /// @brief Returns the nmi value
    virtual double GetSimilarityMeasureValue() override { return 0; }
    /// @brief Compute the voxel based nmi gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override {}
};
/* *************************************************************** */
