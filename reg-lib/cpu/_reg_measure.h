/** @file _reg_measure.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embed all measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "_reg_tools.h"

/// @brief Class common to all measure of similarity classes
class reg_measure {
public:
    /// @brief Measure class constructor
    reg_measure() {
        NR_FUNC_CALLED();
    }
    /// @brief Measure class destructor
    virtual ~reg_measure() {}

    /// @brief Set the pointers to be used by the measure object
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
                                   nifti_image *voxelBasedGradBw = nullptr) {
        this->isSymmetric = false;
        this->referenceImage = refImg;
        this->referenceTimePoints = this->referenceImage->nt;
        this->floatingImage = floImg;
        this->referenceMask = refMask;
        this->warpedImage = warpedImg;
        this->warpedGradient = warpedGrad;
        this->voxelBasedGradient = voxelBasedGrad;
        this->localWeightSim = localWeightSim;
        if (floMask != nullptr && warpedImgBw != nullptr && warpedGradBw != nullptr && voxelBasedGradBw != nullptr) {
            this->isSymmetric = true;
            this->floatingMask = floMask;
            this->warpedImageBw = warpedImgBw;
            this->warpedGradientBw = warpedGradBw;
            this->voxelBasedGradientBw = voxelBasedGradBw;
        } else {
            this->floatingMask = nullptr;
            this->warpedImageBw = nullptr;
            this->warpedGradientBw = nullptr;
            this->voxelBasedGradientBw = nullptr;
        }
        NR_FUNC_CALLED();
    }

    /// @brief Returns the forward registration measure of similarity value
    virtual double GetSimilarityMeasureValueFw() = 0;
    /// @brief Returns the backward registration measure of similarity value
    virtual double GetSimilarityMeasureValueBw() = 0;
    /// @brief Returns the registration measure of similarity value
    double GetSimilarityMeasureValue() {  // Do not override
        // Check that all the specified image are of the same datatype
        if (this->referenceImage->datatype != NIFTI_TYPE_FLOAT32 && this->referenceImage->datatype != NIFTI_TYPE_FLOAT64)
            NR_FATAL_ERROR("Input images are expected to be of floating precision type");
        if (this->warpedImage->datatype != this->referenceImage->datatype)
            NR_FATAL_ERROR("Both input images are expected to have the same type");
        double sim = GetSimilarityMeasureValueFw();
        if (this->isSymmetric) {
            // Check that all the specified image are of the same datatype
            if (this->floatingImage->datatype != NIFTI_TYPE_FLOAT32 && this->floatingImage->datatype != NIFTI_TYPE_FLOAT64)
                NR_FATAL_ERROR("Input images are expected to be of floating precision type");
            if (this->floatingImage->datatype != this->warpedImageBw->datatype)
                NR_FATAL_ERROR("Both input images are expected to have the same type");
            sim += GetSimilarityMeasureValueBw();
        }
        NR_FUNC_CALLED();
        return sim;
    }

    /// @brief Compute the forward voxel-based measure of similarity gradient
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) = 0;
    /// @brief Compute the backward voxel-based measure of similarity gradient
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) = 0;
    /// @brief Compute the voxel-based measure of similarity gradient
    void GetVoxelBasedSimilarityMeasureGradient(int currentTimePoint) {  // Do not override
        // Check if the specified time point exists and is active
        if (currentTimePoint < 0 || currentTimePoint >= this->referenceTimePoints)
            NR_FATAL_ERROR("The specified active time point is not defined in the ref/war images");
        if (this->timePointWeights[currentTimePoint] == 0)
            return;
        // Check if all required input images are of the same data type
        int dtype = this->referenceImage->datatype;
        if (dtype != NIFTI_TYPE_FLOAT32 && dtype != NIFTI_TYPE_FLOAT64)
            NR_FATAL_ERROR("Input images are expected to be of floating precision type");
        if (this->warpedImage->datatype != dtype ||
            this->warpedGradient->datatype != dtype ||
            this->voxelBasedGradient->datatype != dtype)
            NR_FATAL_ERROR("Input images are expected to be of the same type");
        // Compute the gradient
        GetVoxelBasedSimilarityMeasureGradientFw(currentTimePoint);
        if (this->isSymmetric) {
            dtype = this->floatingImage->datatype;
            if (dtype != NIFTI_TYPE_FLOAT32 && dtype != NIFTI_TYPE_FLOAT64)
                NR_FATAL_ERROR("Input images are expected to be of floating precision type");
            if (this->warpedImageBw->datatype != dtype ||
                this->warpedGradientBw->datatype != dtype ||
                this->voxelBasedGradientBw->datatype != dtype)
                NR_FATAL_ERROR("Input images are expected to be of the same type");
            GetVoxelBasedSimilarityMeasureGradientBw(currentTimePoint);
        }
        NR_FUNC_CALLED();
    }
    virtual void GetDiscretisedValue(nifti_image*, float*, int, int) {}
    virtual void SetTimePointWeight(int timePoint, double weight) {
        this->timePointWeights[timePoint] = weight;
    }
    virtual double* GetTimePointWeights() {
        return this->timePointWeights;
    }
    virtual nifti_image* GetReferenceImage() {
        return this->referenceImage;
    }
    virtual int* GetReferenceMask() {
        return this->referenceMask;
    }

protected:
    nifti_image *referenceImage;
    int *referenceMask;
    nifti_image *warpedImage;
    nifti_image *warpedGradient;
    nifti_image *voxelBasedGradient;
    nifti_image *localWeightSim;

    bool isSymmetric;
    nifti_image *floatingImage;
    int *floatingMask;
    nifti_image *warpedImageBw;
    nifti_image *warpedGradientBw;
    nifti_image *voxelBasedGradientBw;

    double timePointWeights[255]{};
    int referenceTimePoints;
};
