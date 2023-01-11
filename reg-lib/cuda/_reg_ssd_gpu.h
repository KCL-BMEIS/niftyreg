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

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief SSD measure of similarity class on the device
class reg_ssd_gpu: public reg_ssd, public reg_measure_gpu {
public:
    /// @brief reg_ssd class constructor
    reg_ssd_gpu();
    /// @brief Measure class destructor
    virtual ~reg_ssd_gpu() {}
    /// @brief Initialise the reg_ssd object
    void InitialiseMeasure(nifti_image *refImgPtr,
                           nifti_image *floImgPtr,
                           int *maskRefPtr,
                           int activeVoxNum,
                           nifti_image *warFloImgPtr,
                           nifti_image *warFloGraPtr,
                           nifti_image *forVoxBasedGraPtr,
                           nifti_image *localWeightSimPtr,
                           cudaArray *refDevicePtr,
                           cudaArray *floDevicePtr,
                           int *refMskDevicePtr,
                           float *warFloDevicePtr,
                           float4 *warFloGradDevicePtr,
                           float4 *forVoxBasedGraDevicePtr);
    /// @brief Returns the ssd value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based ssd gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override;
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
extern "C++"
float reg_getSSDValue_gpu(nifti_image *referenceImage,
                          cudaArray **reference_d,
                          float **warped_d,
                          int **mask_d,
                          int activeVoxelNumber);
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
extern "C++"
void reg_getVoxelBasedSSDGradient_gpu(nifti_image *referenceImage,
                                      cudaArray *reference_d,
                                      float *warped_d,
                                      float4 *spaGradient_d,
                                      float4 *ssdGradient_d,
                                      float maxSD,
                                      int *mask_d,
                                      int activeVoxelNumber);
