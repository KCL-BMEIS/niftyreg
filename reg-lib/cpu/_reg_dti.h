/**
 * @file _reg_dti.h
 * @brief File that contains sum squared difference related function
 * @author Marc Modat
 * @date 19/05/2009
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_ssd.h"

/* *************************************************************** */
/// @brief DTI related measure of similarity class
class reg_dti: public reg_measure {
public:
    /// @brief reg_dti class constructor
    reg_dti();
    /// @brief reg_dti class destructor
    virtual ~reg_dti() {}

    /// @brief Initialise the reg_dti object
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
    /// @brief Returns the dti value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the dti value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based gradient for DTI images forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based gradient for DTI images backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;

protected:
    // Store the indicies of the DT components in the order XX,XY,YY,XZ,YZ,ZZ
    unsigned dtIndicies[6];
    float currentValue;
};
/* *************************************************************** */
/** @brief Computes and returns the SSD between two input image
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to nullptr, all voxels are considered
 * @return Returns an L2 measure of the distance between the anisotropic components of the diffusion tensors
 */
template <class DataType>
double reg_getDtiMeasureValue(const nifti_image *referenceImage,
                              const nifti_image *warpedImage,
                              const int *mask,
                              const unsigned *dtIndicies);
/* *************************************************************** */
/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param warpedGradient Spatial gradient of the input warped image
 * @param dtiMeasureGradientImage Output image that will be updated with the
 * value of the dti measure gradient
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to nullptr, all voxels are considered
 */
template <class DataType>
void reg_getVoxelBasedDtiMeasureGradient(nifti_image *referenceImage,
                                         nifti_image *warpedImage,
                                         nifti_image *warpedGradient,
                                         nifti_image *dtiMeasureGradientImage,
                                         int *mask,
                                         unsigned *dtIndicies);
/* *************************************************************** */
