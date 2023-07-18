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
    /// @brief Returns the lncc value
    virtual double GetSimilarityMeasureValue() override;
    /// @brief Compute the voxel based lncc gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override;
    /// @brief Stuff
    virtual void SetKernelStandardDeviation(int t, float stddev) {
        this->kernelStandardDeviation[t] = stddev;
    }
    /// @brief Stuff
    virtual void SetKernelType(int t) {
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

    int kernelType;

    template <class DataType>
    void UpdateLocalStatImages(nifti_image *refImage,
                               nifti_image *warImage,
                               nifti_image *meanImage,
                               nifti_image *warpedMeanImage,
                               nifti_image *stdDevImage,
                               nifti_image *warpedSdevImage,
                               int *refMask,
                               int *mask,
                               int currentTimepoint);
};
/* *************************************************************** */
/** @brief Compute and return the LNCC between two input image
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 * to use.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to nullptr, all voxels are considered
 * @return Returns the computed LNCC
 */
extern "C++" template<class DataType>
double reg_getLNCCValue(nifti_image *referenceImage,
                        nifti_image *meanImage,
                        nifti_image *sdevImage,
                        nifti_image *warpedImage,
                        nifti_image *warpedMeanImage,
                        nifti_image *warpedSdevImage,
                        int *combinedMask,
                        float *kernelStandardDeviation,
                        nifti_image *correlationImage,
                        int kernelType,
                        int currentTimepoint);
/* *************************************************************** */
/** @brief Compute a voxel based gradient of the LNCC.
 *  @param referenceImage First input image to use to compute the metric
 *  @param warpedImage Second input image to use to compute the metric
 *  @param warpedImageGradient Spatial gradient of the input warped image
 *  @param lnccGradientImage Output image that will be updated with the
 *  value of the LNCC gradient
 *  @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 *  to use.
 *  @param mask Array that contains a mask to specify which voxel
 *  should be considered. If set to nullptr, all voxels are considered
 */
extern "C++" template <class DataType>
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *meanImage,
                                   nifti_image *sdevImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedStdDevImage,
                                   int *combinedMask,
                                   float *kernelStdDev,
                                   nifti_image *correlationImage,
                                   nifti_image *warpedGradient,
                                   nifti_image *lnccGradientImage,
                                   int kernelType,
                                   int currentTimepoint,
                                   double timepointWeight);
/* *************************************************************** */
