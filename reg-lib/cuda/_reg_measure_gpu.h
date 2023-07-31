/** @file _reg_measure_gpu.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embed all gpu measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "_reg_lncc.h"
#include "_reg_dti.h"
#include "_reg_common_cuda.h"
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
                                   float4 *voxelBasedGradBwCuda = nullptr) {
        // Check that the input image are of type float
        if (refImg->datatype != NIFTI_TYPE_FLOAT32 || warpedImg->datatype != NIFTI_TYPE_FLOAT32) {
            reg_print_fct_error("reg_measure_gpu::InitialiseMeasure");
            reg_print_msg_error("Only single precision is supported on the GPU");
            reg_exit();
        }
        // Bind the required pointers
        this->referenceImageCuda = refImgCuda;
        this->floatingImageCuda = floImgCuda;
        this->referenceMaskCuda = refMaskCuda;
        this->activeVoxelNumber = activeVoxNum;
        this->warpedImageCuda = warpedImgCuda;
        this->warpedGradientCuda = warpedGradCuda;
        this->voxelBasedGradientCuda = voxelBasedGradCuda;
        // Check if the symmetric mode is used
        if (floMask != nullptr && warpedImgBw != nullptr && warpedGradBw != nullptr && voxelBasedGradBw != nullptr &&
            floMaskCuda != nullptr && warpedImgBwCuda != nullptr && warpedGradBwCuda != nullptr && voxelBasedGradBwCuda != nullptr) {
            if (floImg->datatype != NIFTI_TYPE_FLOAT32 || warpedImgBw->datatype != NIFTI_TYPE_FLOAT32) {
                reg_print_fct_error("reg_measure_gpu::InitialiseMeasure");
                reg_print_msg_error("Only single precision is supported on the GPU");
                reg_exit();
            }
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
#ifndef NDEBUG
        reg_print_msg_debug("reg_measure_gpu::InitialiseMeasure() called");
#endif
    }

protected:
    cudaArray *referenceImageCuda;
    cudaArray *floatingImageCuda;
    int *referenceMaskCuda;
    size_t activeVoxelNumber;
    float *warpedImageCuda;
    float4 *warpedGradientCuda;
    float4 *voxelBasedGradientCuda;

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
        reg_print_fct_error("reg_lncc_gpu::reg_lncc_gpu");
        reg_print_msg_error("CUDA CANNOT BE USED WITH LNCC YET");
        reg_exit();
    }
    /// @brief reg_lncc class destructor
    virtual ~reg_lncc_gpu() {}

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
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the lncc value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the lncc value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel based lncc gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override {}
};
/* *************************************************************** */
class reg_kld_gpu: public reg_kld, public reg_measure_gpu {
public:
    /// @brief reg_kld_gpu class constructor
    reg_kld_gpu() {
        reg_print_fct_error("reg_kld_gpu::reg_kld_gpu");
        reg_print_msg_error("CUDA CANNOT BE USED WITH KLD YET");
        reg_exit();
    }
    /// @brief reg_kld_gpu class destructor
    virtual ~reg_kld_gpu() {}

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
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the kld value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the kld value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel based kld gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override {}
};
/* *************************************************************** */
class reg_dti_gpu: public reg_dti, public reg_measure_gpu {
public:
    /// @brief reg_dti_gpu class constructor
    reg_dti_gpu() {
        reg_print_fct_error("reg_dti_gpu::reg_dti_gpu");
        reg_print_msg_error("CUDA CANNOT BE USED WITH DTI YET");
        reg_exit();
    }
    /// @brief reg_dti_gpu class destructor
    virtual ~reg_dti_gpu() {}

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
                                   float4 *voxelBasedGradBwCuda = nullptr) override {}
    /// @brief Returns the dti value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the dti value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }
    /// @brief Compute the voxel based dti gradient
    virtual void GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) override {}
};
/* *************************************************************** */
