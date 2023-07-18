/** @file _reg_measure.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embed all measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "_reg_tools.h"
#include <time.h>

/// @brief Class common to all measure of similarity classes
class reg_measure {
public:
    /// @brief Measure class constructor
    reg_measure() {
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] reg_measure constructor called\n");
#endif
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
        this->referenceTimePoint = this->referenceImage->nt;
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
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] reg_measure::InitialiseMeasure()\n");
#endif
    }

    /// @brief Returns the registration measure of similarity value
    virtual double GetSimilarityMeasureValue() = 0;

    /// @brief Compute the voxel based measure of similarity gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
        if (currentTimepoint < 0 || currentTimepoint >= this->referenceImage->nt) {
            reg_print_fct_error("reg_measure::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
            reg_exit();
        }
    }
    virtual void GetDiscretisedValue(nifti_image *, float *, int, int) {}
    virtual void SetTimepointWeight(int timepoint, double weight) {
        this->timePointWeight[timepoint] = weight;
    }
    virtual double* GetTimepointsWeights(void) {
        return this->timePointWeight;
    }
    virtual nifti_image* GetReferenceImage(void) {
        return this->referenceImage;
    }
    virtual int* GetReferenceMask(void) {
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

    double timePointWeight[255] = {0};
    int referenceTimePoint;
};
