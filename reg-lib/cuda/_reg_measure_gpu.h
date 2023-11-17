/** @file _reg_measure_gpu.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embed all gpu measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "CudaCommon.hpp"
#include "_reg_lncc.h"
#include "_reg_dti.h"
#include "_reg_kld.h"

/* *************************************************************** */
/// @brief Class that contains the GPU device pointers
class reg_measure_gpu {
public:
    /// @brief Measure class constructor
    reg_measure_gpu() {}
    /// @brief Measure class destructor
    virtual ~reg_measure_gpu() {}

    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
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
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) {
        // Check that the input image are of type float
        if (refImg->datatype != NIFTI_TYPE_FLOAT32 || warpedImg->datatype != NIFTI_TYPE_FLOAT32)
            NR_FATAL_ERROR("Only single precision is supported on the GPU");
        // Bind the required pointers
        this->referenceImageCuda = refImgCuda;
        this->floatingImageCuda = floImgCuda;
        this->referenceMaskCuda = refMaskCuda;
        this->activeVoxelNumber = activeVoxNum;
        this->warpedImageCuda = warpedImgCuda;
        this->warpedGradientCuda = warpedGradCuda;
        this->voxelBasedGradientCuda = voxelBasedGradCuda;
        this->localWeightSimCuda = localWeightSimCuda;
        // Check if the symmetric mode is used
        if (floMask != nullptr && warpedImgBw != nullptr && warpedGradBw != nullptr && voxelBasedGradBw != nullptr &&
            floMaskCuda != nullptr && warpedImgBwCuda != nullptr && warpedGradBwCuda != nullptr && voxelBasedGradBwCuda != nullptr) {
            if (floImg->datatype != NIFTI_TYPE_FLOAT32 || warpedImgBw->datatype != NIFTI_TYPE_FLOAT32)
                NR_FATAL_ERROR("Only single precision is supported on the GPU");
            this->floatingMaskCuda = floMaskCuda;
            this->warpedImageBwCuda = warpedImgBwCuda;
            this->warpedGradientBwCuda = warpedGradBwCuda;
            this->voxelBasedGradientBwCuda = voxelBasedGradBwCuda;
        } else {
            this->floatingMaskCuda = nullptr;
            this->warpedImageBwCuda = nullptr;
            this->warpedGradientBwCuda = nullptr;
            this->voxelBasedGradientBwCuda = nullptr;
        }
        NR_FUNC_CALLED();
    }

protected:
    float *referenceImageCuda;
    float *floatingImageCuda;
    int *referenceMaskCuda;
    size_t activeVoxelNumber;
    float *warpedImageCuda;
    float4 *warpedGradientCuda;
    float4 *voxelBasedGradientCuda;
    float *localWeightSimCuda;

    int *floatingMaskCuda;
    float *warpedImageBwCuda;
    float4 *warpedGradientBwCuda;
    float4 *voxelBasedGradientBwCuda;
};
/* *************************************************************** */
class reg_lncc_gpu: public reg_lncc, public reg_measure_gpu {
public:
    /// @brief reg_lncc class constructor
    reg_lncc_gpu() {
        NR_FATAL_ERROR("CUDA CANNOT BE USED WITH LNCC YET");
    }
    /// @brief reg_lncc class destructor
    virtual ~reg_lncc_gpu() {}

    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
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
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the lncc value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the lncc value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel-based lncc gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override {}
    /// @brief Compute the voxel-based lncc gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override {}
};
/* *************************************************************** */
class reg_kld_gpu: public reg_kld, public reg_measure_gpu {
public:
    /// @brief reg_kld_gpu class constructor
    reg_kld_gpu() {
        NR_FATAL_ERROR("CUDA CANNOT BE USED WITH KLD YET");
    }
    /// @brief reg_kld_gpu class destructor
    virtual ~reg_kld_gpu() {}

    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
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
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the kld value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the kld value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel-based kld gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override {}
    /// @brief Compute the voxel-based kld gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override {}
};
/* *************************************************************** */
class reg_dti_gpu: public reg_dti, public reg_measure_gpu {
public:
    /// @brief reg_dti_gpu class constructor
    reg_dti_gpu() {
        NR_FATAL_ERROR("CUDA CANNOT BE USED WITH DTI YET");
    }
    /// @brief reg_dti_gpu class destructor
    virtual ~reg_dti_gpu() {}

    virtual void InitialiseMeasure(nifti_image *refImg,
                                   float *refImgCuda,
                                   nifti_image *floImg,
                                   float *floImgCuda,
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
                                   float *localWeightSimCuda = nullptr,
                                   int *floMask = nullptr,
                                   int *floMaskCuda = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   float *warpedImgBwCuda = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   float4 *warpedGradBwCuda = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr,
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the dti value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the dti value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel-based dti gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override {}
    /// @brief Compute the voxel-based dti gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override {}
};
/* *************************************************************** */
