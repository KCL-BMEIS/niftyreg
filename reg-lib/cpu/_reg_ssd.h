/**
 * @file _reg_ssd.h
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

#include "_reg_measure.h"

/* *************************************************************** */
/// @brief SSD measure of similarity class
class reg_ssd: public reg_measure {
public:
    /// @brief reg_ssd class constructor
    reg_ssd();
    /// @brief reg_ssd class destructor
    virtual ~reg_ssd() {}

    /// @brief Initialise the reg_ssd object
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
    /// @brief Define if the specified time point should be normalised
    void SetNormaliseTimePoint(int timePoint, bool normalise);
    /// @brief Returns the ssd value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the ssd value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based ssd gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based ssd gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;
    /// @brief Here
    virtual void GetDiscretisedValue(nifti_image *controlPointGridImage,
                                     float *discretisedValue,
                                     int discretiseRadius,
                                     int discretiseStep) override;

protected:
    bool normaliseTimePoint[255];
};
/* *************************************************************** */
/** @brief Computes and returns the SSD between two input images
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param timePointWeights Array that contains the weight of each time point
 * @param jacobianDetImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SSD. The argument is ignored if the
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered
 * @param localWeightSim Image that contains the local weight similarity
 * @return Returns the computed sum squared difference
 */
template <class DataType>
double reg_getSsdValue(const nifti_image *referenceImage,
                       const nifti_image *warpedImage,
                       const double *timePointWeights,
                       const int referenceTimePoints,
                       const nifti_image *jacobianDetImage,
                       const int *mask,
                       const nifti_image *localWeightSim);
/* *************************************************************** */
/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param warpedGradient Spatial gradient of the input warped image
 * @param measureGradientImage Output image that will be updated with the
 * value of the SSD gradient
 * @param jacobianDetImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SSD. The argument is ignored if the
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered
 * @param currentTimePoint Specifies which time point volumes have to be considered
 * @param timePointWeight Weight of the specified time point
 * @param localWeightSim Image that contains the local weight similarity
 */
template <class DataType>
void reg_getVoxelBasedSsdGradient(const nifti_image *referenceImage,
                                  const nifti_image *warpedImage,
                                  const nifti_image *warpedGradient,
                                  nifti_image *measureGradientImage,
                                  const nifti_image *jacobianDetImage,
                                  const int *mask,
                                  const int currentTimePoint,
                                  const double timePointWeight,
                                  const nifti_image *localWeightSim);
/* *************************************************************** */
