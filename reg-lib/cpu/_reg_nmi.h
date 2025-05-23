/*
 *  _reg_nmi.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_measure.h"

/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_nmi: public reg_measure {
public:
    /// @brief reg_nmi class constructor
    reg_nmi();
    /// @brief reg_nmi class destructor
    virtual ~reg_nmi();

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
    /// @brief Returns the nmi value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the nmi value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based nmi gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    /// @brief Compute the voxel-based nmi gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;

    virtual void SetRefAndFloatBinNumbers(unsigned short refBinNumber,
                                          unsigned short floBinNumber,
                                          int timePoint) {
        this->referenceBinNumber[timePoint] = refBinNumber;
        this->floatingBinNumber[timePoint] = floBinNumber;
    }
    virtual void SetReferenceBinNumber(int b, int t) {
        this->referenceBinNumber[t] = b;
    }
    virtual void SetFloatingBinNumber(int b, int t) {
        this->floatingBinNumber[t] = b;
    }
    virtual unsigned short* GetReferenceBinNumber() {
        return this->referenceBinNumber;
    }
    virtual unsigned short* GetFloatingBinNumber() {
        return this->floatingBinNumber;
    }
    virtual void ApproximatePw() {
        this->approximatePw = true;
    }
    virtual void DoNotApproximatePw() {
        this->approximatePw = false;
    }

protected:
    bool approximatePw = true;
    unsigned short referenceBinNumber[255];
    unsigned short floatingBinNumber[255];
    unsigned short totalBinNumber[255]{};
    double **jointHistogramPro = nullptr;
    double **jointHistogramLog = nullptr;
    double **entropyValues = nullptr;
    double **jointHistogramProBw = nullptr;
    double **jointHistogramLogBw = nullptr;
    double **entropyValuesBw = nullptr;

    void DeallocateHistogram();
};
/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_multichannel_nmi: public reg_measure {
public:
    /// @brief reg_multichannel_nmi class constructor
    reg_multichannel_nmi() {}
    /// @brief reg_multichannel_nmi class destructor
    virtual ~reg_multichannel_nmi() {}

    /// @brief Returns the nmi value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the nmi value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }

    /// @brief Compute the voxel-based nmi gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override {}
    /// @brief Compute the voxel-based nmi gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override {}

protected:
    unsigned short referenceBinNumber[255]{};
    unsigned short floatingBinNumber[255]{};
    unsigned short totalBinNumber[255]{};
    double *jointHistogramProp = nullptr;
    double *jointHistogramLog = nullptr;
    double *entropyValues = nullptr;
    double *jointHistogramPropBw = nullptr;
    double *jointHistogramLogBw = nullptr;
    double *entropyValuesBw = nullptr;
};
/* *************************************************************** */
/// Multi channel NMI version - Entropy
void reg_getMultiChannelNmiValue(nifti_image *referenceImages,
                                 nifti_image *warpedImages,
                                 unsigned *referenceBins, // should be an array of size num_reference_volumes
                                 unsigned *warpedBins, // should be an array of size num_warped_volumes
                                 double *probaJointHistogram,
                                 double *logJointHistogram,
                                 double *entropies,
                                 int *mask,
                                 bool approx);
/* *************************************************************** */
/// Multi channel NMI version - Gradient
void reg_getVoxelBasedMultiChannelNmiGradient2D(nifti_image *referenceImages,
                                                nifti_image *warpedImages,
                                                nifti_image *warpedImageGradient,
                                                unsigned *referenceBins,
                                                unsigned *warpedBins,
                                                double *logJointHistogram,
                                                double *entropies,
                                                nifti_image *nmiGradientImage,
                                                int *mask,
                                                bool approx);
/* *************************************************************** */
/// Multi channel NMI version - Gradient
void reg_getVoxelBasedMultiChannelNmiGradient3D(nifti_image *referenceImages,
                                                nifti_image *warpedImages,
                                                nifti_image *warpedImageGradient,
                                                unsigned *referenceBins,
                                                unsigned *warpedBins,
                                                double *logJointHistogram,
                                                double *entropies,
                                                nifti_image *nmiGradientImage,
                                                int *mask,
                                                bool approx);
/* *************************************************************** */
template<class PrecisionType>
DEVICE constexpr PrecisionType GetBasisSplineValue(PrecisionType x) {
    x = x < 0 ? -x : x;
    PrecisionType value = 0;
    if (x < 2.f) {
        if (x < 1.f)
            value = 2.f / 3.f + (0.5f * x - 1.f) * x * x;
        else {
            x -= 2.f;
            value = -x * x * x / 6.f;
        }
    }
    return value;
}
/* *************************************************************** */
template<class PrecisionType>
DEVICE constexpr PrecisionType GetBasisSplineDerivativeValue(const PrecisionType origX) {
    PrecisionType x = origX < 0 ? -origX : origX;
    PrecisionType value = 0;
    if (x < 2.f) {
        if (x < 1.f)
            value = (1.5f * x - 2.f) * origX;
        else {
            x -= 2.f;
            value = -0.5f * x * x;
            if (origX < 0) value = -value;
        }
    }
    return value;
}
/* *************************************************************** */
