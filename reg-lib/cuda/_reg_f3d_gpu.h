/*
 *  _reg_f3d_gpu.h
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_resampling_gpu.h"
#include "_reg_globalTransformation_gpu.h"
#include "_reg_localTransformation_gpu.h"
#include "_reg_nmi_gpu.h"
#include "_reg_ssd_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_common_cuda.h"
#include "_reg_optimiser_gpu.h"
#include "_reg_f3d.h"

class reg_f3d_gpu: public reg_f3d<float> {
protected:
    // cuda variables
    cudaArray *reference_gpu;
    cudaArray *floating_gpu;
    int *currentMask_gpu;
    float *warped_gpu;
    float4 *controlPointGrid_gpu;
    float4 *deformationFieldImage_gpu;
    float4 *warpedGradientImage_gpu;
    float4 *voxelBasedMeasureGradientImage_gpu;
    float4 *transformationGradient_gpu;

    // cuda variable for multispectral registration
    cudaArray *reference2_gpu;
    cudaArray *floating2_gpu;
    float *warped2_gpu;
    float4 *warpedGradientImage2_gpu;

    // Measure related variables
    reg_ssd_gpu *measure_gpu_ssd;
    reg_kld_gpu *measure_gpu_kld;
    reg_dti_gpu *measure_gpu_dti;
    reg_lncc_gpu *measure_gpu_lncc;
    reg_nmi_gpu *measure_gpu_nmi;

    float InitialiseCurrentLevel();
    void DeallocateCurrentInputImage();
    void AllocateWarped();
    void DeallocateWarped();
    void AllocateDeformationField();
    void DeallocateDeformationField();
    void AllocateWarpedGradient();
    void DeallocateWarpedGradient();
    void AllocateVoxelBasedMeasureGradient();
    void DeallocateVoxelBasedMeasureGradient();
    void AllocateTransformationGradient();
    void DeallocateTransformationGradient();

    double ComputeJacobianBasedPenaltyTerm(int);
    double ComputeBendingEnergyPenaltyTerm();
    double ComputeLinearEnergyPenaltyTerm();
    double ComputeLandmarkDistancePenaltyTerm();
    void GetDeformationField();
    void WarpFloatingImage(int);
    void GetVoxelBasedGradient();
    void GetSimilarityMeasureGradient();
    void GetBendingEnergyGradient();
    void GetLinearEnergyGradient();
    void GetJacobianBasedGradient();
    void GetLandmarkDistanceGradient();
    void SmoothGradient();
    void GetApproximatedGradient();
    void UpdateParameters(float);
    void SetOptimiser();
    // void SetGradientImageToZero();
    float NormaliseGradient();
    void InitialiseSimilarity();

public:
    void UseNMISetReferenceBinNumber(int, int);
    void UseNMISetFloatingBinNumber(int, int);
    void UseSSD(int timepoint);
    void UseKLDivergence(int timepoint);
    void UseDTI(int timepoint[6]);
    void UseLNCC(int timepoint, float stdDevKernel);
    nifti_image** GetWarpedImage();

    reg_f3d_gpu(int refTimePoint, int floTimePoint);
    ~reg_f3d_gpu();
    int CheckMemoryMB();
};

#include "_reg_f3d_gpu.cpp"
