/*
 * @file _reg_lncc_gpu.h
 * @brief CUDA implementation of the LNCC similarity measure
 * Copyright (c) 2026, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

#include "CudaTools.hpp"
#include "_reg_measure_gpu.h"
#include "_reg_lncc.h"

/* *************************************************************** */
class reg_lncc_gpu: public reg_lncc, public reg_measure_gpu {
public:
    reg_lncc_gpu();
    virtual ~reg_lncc_gpu();

    // Bring the CPU base overload into scope; the GPU override below intentionally adds a second overload
    using reg_measure::InitialiseMeasure;
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
                                   float4 *voxelBasedGradBwCuda = nullptr) override;

    virtual double GetSimilarityMeasureValueFw() override;
    virtual double GetSimilarityMeasureValueBw() override;
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) override;
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) override;

private:
    // Forward pass intermediate buffers (float4, using .x component for single-channel values)
    thrust::device_vector<float4> meanBuf;
    thrust::device_vector<float4> sdevBuf;
    thrust::device_vector<float4> warpedMeanBuf;
    thrust::device_vector<float4> warpedSdevBuf;
    thrust::device_vector<float4> correlationBuf;

    // Backward pass intermediate buffers (symmetric mode)
    thrust::device_vector<float4> meanBufBw;
    thrust::device_vector<float4> sdevBufBw;
    thrust::device_vector<float4> warpedMeanBufBw;
    thrust::device_vector<float4> warpedSdevBufBw;
    thrust::device_vector<float4> correlationBufBw;

    // Full per-voxel registration masks uploaded from CPU (-1 = excluded)
    thrust::device_vector<int> refMaskFullCuda;
    thrust::device_vector<int> floMaskFullCuda;
};
/* *************************************************************** */
