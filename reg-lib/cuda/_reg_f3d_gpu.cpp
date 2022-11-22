/*
 *  _reg_f3d_gpu.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_GPU_CPP
#define _REG_F3D_GPU_CPP

#include "_reg_f3d_gpu.h"

 /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
 /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::reg_f3d_gpu(int refTimePoint, int floTimePoint)
    : reg_f3d<float>::reg_f3d(refTimePoint, floTimePoint) {
    this->executableName = (char *)"NiftyReg F3D GPU";
    this->currentReference_gpu = NULL;
    this->currentFloating_gpu = NULL;
    this->currentMask_gpu = NULL;
    this->warped_gpu = NULL;
    this->controlPointGrid_gpu = NULL;
    this->deformationFieldImage_gpu = NULL;
    this->warpedGradientImage_gpu = NULL;
    this->voxelBasedMeasureGradientImage_gpu = NULL;
    this->transformationGradient_gpu = NULL;

    this->measure_gpu_ssd = NULL;
    this->measure_gpu_kld = NULL;
    this->measure_gpu_dti = NULL;
    this->measure_gpu_lncc = NULL;
    this->measure_gpu_nmi = NULL;

    this->currentReference2_gpu = NULL;
    this->currentFloating2_gpu = NULL;
    this->warped2_gpu = NULL;
    this->warpedGradientImage2_gpu = NULL;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::reg_f3d_gpu");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::~reg_f3d_gpu() {
    if (this->currentReference_gpu != NULL)
        cudaCommon_free(&this->currentReference_gpu);
    if (this->currentFloating_gpu != NULL)
        cudaCommon_free(&this->currentFloating_gpu);
    if (this->currentMask_gpu != NULL)
        cudaCommon_free<int>(&this->currentMask_gpu);
    if (this->warped_gpu != NULL)
        cudaCommon_free<float>(&this->warped_gpu);
    if (this->controlPointGrid_gpu != NULL)
        cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    if (this->deformationFieldImage_gpu != NULL)
        cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
    if (this->warpedGradientImage_gpu != NULL)
        cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
    if (this->voxelBasedMeasureGradientImage_gpu != NULL)
        cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
    if (this->transformationGradient_gpu != NULL)
        cudaCommon_free<float4>(&this->transformationGradient_gpu);

    if (this->currentReference2_gpu != NULL)
        cudaCommon_free(&this->currentReference2_gpu);
    if (this->currentFloating2_gpu != NULL)
        cudaCommon_free(&this->currentFloating2_gpu);
    if (this->warped2_gpu != NULL)
        cudaCommon_free<float>(&this->warped2_gpu);
    if (this->warpedGradientImage2_gpu != NULL)
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);

    if (this->optimiser != NULL) {
        delete this->optimiser;
        this->optimiser = NULL;
    }

    if (this->measure_gpu_nmi != NULL) {
        delete this->measure_gpu_nmi;
        this->measure_gpu_nmi = NULL;
        this->measure_nmi = NULL;
    }
    if (this->measure_gpu_ssd != NULL) {
        delete this->measure_gpu_ssd;
        this->measure_gpu_ssd = NULL;
        this->measure_ssd = NULL;
    }
    if (this->measure_gpu_kld != NULL) {
        delete this->measure_gpu_kld;
        this->measure_gpu_kld = NULL;
        this->measure_kld = NULL;
    }
    if (this->measure_gpu_dti != NULL) {
        delete this->measure_gpu_dti;
        this->measure_gpu_dti = NULL;
        this->measure_dti = NULL;
    }
    if (this->measure_gpu_lncc != NULL) {
        delete this->measure_gpu_lncc;
        this->measure_gpu_lncc = NULL;
        this->measure_lncc = NULL;
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::~reg_f3d_gpu");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateWarped() {
    reg_f3d::AllocateWarped();

    if (this->warped->nt == 1) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, this->warped->dim)) {
            reg_print_fct_error("reg_f3d_gpu::AllocateWarped()");
            reg_print_msg_error("Error when allocating the warped image");
            reg_exit();
        }
    } else if (this->warped->nt == 2) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, &this->warped2_gpu, this->warped->dim)) {
            reg_print_fct_error("reg_f3d_gpu::AllocateWarped()");
            reg_print_msg_error("Error when allocating the warped image");
            reg_exit();
        }
    } else {
        reg_print_fct_error("reg_f3d_gpu::AllocateWarped()");
        reg_print_msg_error("reg_f3d_gpu does not handle more than 2 time points in the floating image");
        reg_exit();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateWarped");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearWarped() {
    reg_f3d::ClearWarped();

    if (this->warped_gpu != NULL) {
        cudaCommon_free<float>(&this->warped_gpu);
        this->warped_gpu = NULL;
    }
    if (this->warped2_gpu != NULL) {
        cudaCommon_free<float>(&this->warped2_gpu);
        this->warped2_gpu = NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearWarped");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateDeformationField() {
    this->ClearDeformationField();
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->deformationFieldImage_gpu,
                                 this->activeVoxelNumber[this->currentLevel] * sizeof(float4)));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearDeformationField() {
    if (this->deformationFieldImage_gpu != NULL) {
        cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
        this->deformationFieldImage_gpu = NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateWarpedGradient() {
    this->ClearWarpedGradient();
    if (this->inputFloating->nt == 1) {
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                     this->activeVoxelNumber[this->currentLevel] * sizeof(float4)));
    } else if (this->inputFloating->nt == 2) {
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                     this->activeVoxelNumber[this->currentLevel] * sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage2_gpu,
                                     this->activeVoxelNumber[this->currentLevel] * sizeof(float4)));
    } else {
        reg_print_fct_error("reg_f3d_gpu::AllocateWarpedGradient()");
        reg_print_msg_error("reg_f3d_gpu does not handle more than 2 time points in the floating image");
        reg_exit();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateWarpedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearWarpedGradient() {
    if (this->warpedGradientImage_gpu != NULL) {
        cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
        this->warpedGradientImage_gpu = NULL;
    }
    if (this->warpedGradientImage2_gpu != NULL) {
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);
        this->warpedGradientImage2_gpu = NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearWarpedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateVoxelBasedMeasureGradient() {
    this->ClearVoxelBasedMeasureGradient();
    if (cudaCommon_allocateArrayToDevice(&this->voxelBasedMeasureGradientImage_gpu, this->currentReference->dim)) {
        reg_print_fct_error("reg_f3d_gpu::AllocateVoxelBasedMeasureGradient()");
        reg_print_msg_error("Error when allocating the voxel based measure gradient image");
        reg_exit();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateVoxelBasedMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearVoxelBasedMeasureGradient() {
    if (this->voxelBasedMeasureGradientImage_gpu != NULL) {
        cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
        this->voxelBasedMeasureGradientImage_gpu = NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearVoxelBasedMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateTransformationGradient() {
    this->ClearTransformationGradient();
    if (cudaCommon_allocateArrayToDevice(&this->transformationGradient_gpu, this->controlPointGrid->dim)) {
        reg_print_fct_error("reg_f3d_gpu::AllocateTransformationGradient()");
        reg_print_msg_error("Error when allocating the node based gradient image");
        reg_exit();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateNodeBasedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearTransformationGradient() {
    if (this->transformationGradient_gpu != NULL) {
        cudaCommon_free<float4>(&this->transformationGradient_gpu);
        this->transformationGradient_gpu = NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearTransformationGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeJacobianBasedPenaltyTerm(int type) {
    if (this->jacobianLogWeight <= 0) return 0;

    bool approx = type == 2 ? false : this->jacobianLogApproximation;

    double value = reg_spline_getJacobianPenaltyTerm_gpu(this->currentReference,
                                                         this->controlPointGrid,
                                                         &this->controlPointGrid_gpu,
                                                         approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (value != value && it < maxit) {
        value = reg_spline_correctFolding_gpu(this->currentReference,
                                              this->controlPointGrid,
                                              &this->controlPointGrid_gpu,
                                              approx);
#ifndef NDEBUG
        reg_print_msg_debug("Folding correction");
#endif
        it++;
    }
    if (type > 0) {
        if (value != value) {
            this->optimiser->RestoreBestDOF();
            reg_print_fct_error("reg_f3d_gpu::ComputeJacobianBasedPenaltyTerm()");
            reg_print_msg_error("The folding correction scheme failed");
        } else {
#ifndef NDEBUG
            if (it > 0) {
                char text[255];
                sprintf(text, "Folding correction, %i step(s)", it);
                reg_print_msg_debug(text);
            }
#endif
        }
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ComputeJacobianBasedPenaltyTerm");
#endif
    return this->jacobianLogWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeBendingEnergyPenaltyTerm() {
    if (this->bendingEnergyWeight <= 0) return 0;

    // CHECKED: Similar output
    double value = reg_spline_approxBendingEnergy_gpu(this->controlPointGrid,
                                                      &this->controlPointGrid_gpu);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ComputeBendingEnergyPenaltyTerm");
#endif
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeLinearEnergyPenaltyTerm() {
    if (this->linearEnergyWeight <= 0)
        return 0;

    reg_print_fct_error("reg_f3d_gpu::ComputeLinearEnergyPenaltyTerm()");
    reg_print_msg_error("Option not supported!");
    reg_exit();
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeLandmarkDistancePenaltyTerm() {
    if (this->landmarkRegWeight <= 0)
        return 0;

    reg_print_fct_error("reg_f3d_gpu::ComputeLandmarkDistancePenaltyTerm()");
    reg_print_msg_error("Option not supported!");
    reg_exit();
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetDeformationField() {
    if (this->controlPointGrid_gpu == NULL) {
        reg_f3d::GetDeformationField();
    } else {
        // Compute the deformation field
        reg_spline_getDeformationField_gpu(this->controlPointGrid,
                                           this->currentReference,
                                           &this->controlPointGrid_gpu,
                                           &this->deformationFieldImage_gpu,
                                           &this->currentMask_gpu,
                                           this->activeVoxelNumber[this->currentLevel],
                                           true); // use B-splines
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::WarpFloatingImage(int inter) {
    // Interpolation is linear by default when using GPU, the inter variable is not used.
    inter = inter; // just to avoid a compiler warning

    // Compute the deformation field
    this->GetDeformationField();

    // Resample the floating image
    reg_resampleImage_gpu(this->currentFloating,
                          &this->warped_gpu,
                          &this->currentFloating_gpu,
                          &this->deformationFieldImage_gpu,
                          &this->currentMask_gpu,
                          this->activeVoxelNumber[this->currentLevel],
                          this->warpedPaddingValue);

    if (this->currentFloating->nt == 2) {
        reg_resampleImage_gpu(this->currentFloating,
                              &this->warped2_gpu,
                              &this->currentFloating2_gpu,
                              &this->deformationFieldImage_gpu,
                              &this->currentMask_gpu,
                              this->activeVoxelNumber[this->currentLevel],
                              this->warpedPaddingValue);
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::WarpFloatingImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::SetGradientImageToZero() {
    cudaMemset(this->transformationGradient_gpu, 0,
               this->controlPointGrid->nx * this->controlPointGrid->ny * this->controlPointGrid->nz * sizeof(float4));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::SetGradientImageToZero");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetVoxelBasedGradient() {
    // The voxel based gradient image is filled with zeros
    cudaMemset(this->voxelBasedMeasureGradientImage_gpu, 0,
               this->currentReference->nx * this->currentReference->ny * this->currentReference->nz *
               sizeof(float4));

    // The intensity gradient is first computed
    reg_getImageGradient_gpu(this->currentFloating,
                             &this->currentFloating_gpu,
                             &this->deformationFieldImage_gpu,
                             &this->warpedGradientImage_gpu,
                             this->activeVoxelNumber[this->currentLevel],
                             this->warpedPaddingValue);

    // The gradient of the various measures of similarity are computed
    if (this->measure_gpu_nmi != NULL)
        this->measure_gpu_nmi->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_ssd != NULL)
        this->measure_gpu_ssd->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_kld != NULL)
        this->measure_gpu_kld->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_lncc != NULL)
        this->measure_gpu_lncc->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_dti != NULL)
        this->measure_gpu_dti->GetVoxelBasedSimilarityMeasureGradient();

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetVoxelBasedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetSimilarityMeasureGradient() {
    this->GetVoxelBasedGradient();

    // The voxel based gradient is smoothed
    float smoothingRadius[3] = {
        this->controlPointGrid->dx / this->currentReference->dx,
        this->controlPointGrid->dy / this->currentReference->dy,
        this->controlPointGrid->dz / this->currentReference->dz
    };
    reg_smoothImageForCubicSpline_gpu(this->warped,
                                      &this->voxelBasedMeasureGradientImage_gpu,
                                      smoothingRadius);

    // The node gradient is extracted
    reg_voxelCentric2NodeCentric_gpu(this->warped,
                                     this->controlPointGrid,
                                     &this->voxelBasedMeasureGradientImage_gpu,
                                     &this->transformationGradient_gpu,
                                     this->similarityWeight);

    /* The similarity measure gradient is converted from voxel space to real space */
    mat44 *floatingMatrix_xyz = NULL;
    if (this->currentFloating->sform_code > 0)
        floatingMatrix_xyz = &(this->currentFloating->sto_xyz);
    else floatingMatrix_xyz = &(this->currentFloating->qto_xyz);
    reg_convertNMIGradientFromVoxelToRealSpace_gpu(floatingMatrix_xyz,
                                                   this->controlPointGrid,
                                                   &this->transformationGradient_gpu);
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (this->gradientSmoothingSigma != 0) {
        reg_gaussianSmoothing_gpu(this->controlPointGrid,
                                  &this->transformationGradient_gpu,
                                  this->gradientSmoothingSigma,
                                  NULL);
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetSimilarityMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetBendingEnergyGradient() {
    if (this->bendingEnergyWeight <= 0) return;

    reg_spline_approxBendingEnergyGradient_gpu(this->controlPointGrid,
                                               &this->controlPointGrid_gpu,
                                               &this->transformationGradient_gpu,
                                               this->bendingEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetBendingEnergyGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetLinearEnergyGradient() {
    if (this->linearEnergyWeight <= 0)
        return;

    reg_print_fct_error("reg_f3d_gpu::GetLinearEnergyGradient()");
    reg_print_msg_error("Option not supported!");
    reg_exit();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetJacobianBasedGradient() {
    if (this->jacobianLogWeight <= 0) return;

    reg_spline_getJacobianPenaltyTermGradient_gpu(this->currentReference,
                                                  this->controlPointGrid,
                                                  &this->controlPointGrid_gpu,
                                                  &this->transformationGradient_gpu,
                                                  this->jacobianLogWeight,
                                                  this->jacobianLogApproximation);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetJacobianBasedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetLandmarkDistanceGradient() {
    if (this->landmarkRegWeight <= 0)
        return;

    reg_print_fct_error("reg_f3d_gpu::GetLandmarkDistanceGradient()");
    reg_print_msg_error("Option not supported!");
    reg_exit();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UpdateParameters(float scale) {
    float4 *currentDOF = reinterpret_cast<float4*>(this->optimiser->GetCurrentDOF());
    float4 *bestDOF = reinterpret_cast<float4*>(this->optimiser->GetBestDOF());
    float4 *gradient = reinterpret_cast<float4*>(this->optimiser->GetGradient());

    reg_updateControlPointPosition_gpu(this->controlPointGrid, &currentDOF, &bestDOF, &gradient, scale);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UpdateParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::SmoothGradient() {
    if (this->gradientSmoothingSigma != 0) {
        reg_print_fct_error("reg_f3d_gpu::SmoothGradient()");
        reg_print_msg_error("Option not supported!");
        reg_exit();
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetApproximatedGradient() {
    float4 *gridValue, *currentValue, *gradientValue;
    cudaMallocHost(&gridValue, sizeof(float4));
    cudaMallocHost(&currentValue, sizeof(float4));
    cudaMallocHost(&gradientValue, sizeof(float4));

    float eps = this->controlPointGrid->dx / 100.f;

    for (size_t i = 0; i < this->optimiser->GetVoxNumber(); ++i) {
        // Extract the grid value
        cudaMemcpy(gridValue, &this->controlPointGrid_gpu[i], sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(currentValue, &(reinterpret_cast<float4*>(this->optimiser->GetBestDOF()))[i], sizeof(float4), cudaMemcpyDeviceToHost);

        // -- X axis
        // Modify the grid value along the x axis
        gridValue->x = currentValue->x + eps;
        cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
        // Evaluate the objective function value
        gradientValue->x = this->GetObjectiveFunctionValue();
        // Modify the grid value along the x axis
        gridValue->x = currentValue->x - eps;
        cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
        // Evaluate the objective function value
        gradientValue->x -= this->GetObjectiveFunctionValue();
        gradientValue->x /= 2.f * eps;
        gridValue->x = currentValue->x;

        // -- Y axis
        // Modify the grid value along the y axis
        gridValue->y = currentValue->y + eps;
        cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
        // Evaluate the objective function value
        gradientValue->y = this->GetObjectiveFunctionValue();
        // Modify the grid value the y axis
        gridValue->y = currentValue->y - eps;
        cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
        // Evaluate the objective function value
        gradientValue->y -= this->GetObjectiveFunctionValue();
        gradientValue->y /= 2.f * eps;
        gridValue->y = currentValue->y;

        if (this->optimiser->GetNDim() > 2) {
            // -- Z axis
            // Modify the grid value along the y axis
            gridValue->z = currentValue->z + eps;
            cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
            // Evaluate the objective function value
            gradientValue->z = this->GetObjectiveFunctionValue();
            // Modify the grid value the y axis
            gridValue->z = currentValue->z - eps;
            cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);
            // Evaluate the objective function value
            gradientValue->z -= this->GetObjectiveFunctionValue();
            gradientValue->z /= 2.f * eps;
        }

        // Restore the initial parametrisation
        cudaMemcpy(&this->controlPointGrid_gpu[i], gridValue, sizeof(float4), cudaMemcpyHostToDevice);

        // Save the assessed gradient
        cudaMemcpy(&this->transformationGradient_gpu[i], gradientValue, sizeof(float4), cudaMemcpyHostToDevice);
    }

    cudaFreeHost(gridValue);
    cudaFreeHost(currentValue);
    cudaFreeHost(gradientValue);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetApproximatedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::fillImageData(nifti_image *image, float* memoryObject) {
    size_t size = image->nvox;
    float *buffer = (float*)malloc(size * sizeof(float));

    if (buffer == NULL) {
        reg_print_fct_error("reg_f3d_gpu::fillImageData()");
        reg_print_msg_error("Memory allocation did not complete successfully!");
        reg_exit();
    }

    cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

    free(image->data);
    image->datatype = NIFTI_TYPE_FLOAT32;
    image->nbyper = sizeof(float);
    image->data = (void*)malloc(image->nvox * image->nbyper);
    float *dataT = static_cast<float*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = static_cast<float>(buffer[i]);
    free(buffer);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
nifti_image** reg_f3d_gpu::GetWarpedImage() {
    // The initial images are used
    if (this->inputReference == NULL || this->inputFloating == NULL || this->controlPointGrid == NULL) {
        reg_print_fct_error("reg_f3d_gpu::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    this->currentReference = this->inputReference;
    this->currentFloating = this->inputFloating;
    this->currentMask = (int*)calloc(this->activeVoxelNumber[this->currentLevel], sizeof(int));

    reg_tools_changeDatatype<float>(this->currentReference);
    reg_tools_changeDatatype<float>(this->currentFloating);

    this->AllocateWarped();
    this->AllocateDeformationField();
    this->InitialiseCurrentLevel();
    this->WarpFloatingImage(3); // cubic spline interpolation
    this->ClearDeformationField();

    nifti_image **warpedImage = (nifti_image**)calloc(2, sizeof(nifti_image*));
    warpedImage[0] = nifti_copy_nim_info(this->warped);
    warpedImage[0]->cal_min = this->inputFloating->cal_min;
    warpedImage[0]->cal_max = this->inputFloating->cal_max;
    warpedImage[0]->scl_slope = this->inputFloating->scl_slope;
    warpedImage[0]->scl_inter = this->inputFloating->scl_inter;
    this->fillImageData(warpedImage[0], this->warped_gpu);
    if (this->currentFloating->nt == 2)
        this->fillImageData(warpedImage[1], this->warped2_gpu);

    this->ClearWarped();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetWarpedImage");
#endif
    return warpedImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_f3d_gpu::InitialiseCurrentLevel() {
    float maxStepSize = reg_f3d::InitialiseCurrentLevel();

    if (this->currentReference_gpu != NULL) cudaCommon_free(&this->currentReference_gpu);
    if (this->currentReference2_gpu != NULL) cudaCommon_free(&this->currentReference2_gpu);
    if (this->currentReference->nt == 1) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->currentReference_gpu, this->currentReference->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the reference image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentReference_gpu, this->currentReference)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the reference image");
            reg_exit();
        }
    } else if (this->currentReference->nt == 2) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->currentReference_gpu,
                                                    &this->currentReference2_gpu, this->currentReference->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the reference image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentReference_gpu,
                                                           &this->currentReference2_gpu, this->currentReference)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the reference image");
            reg_exit();
        }
    }

    if (this->currentFloating_gpu != NULL) cudaCommon_free(&this->currentFloating_gpu);
    if (this->currentFloating2_gpu != NULL) cudaCommon_free(&this->currentFloating2_gpu);
    if (this->currentReference->nt == 1) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->currentFloating_gpu, this->currentFloating->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the floating image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentFloating_gpu, this->currentFloating)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the floating image");
            reg_exit();
        }
    } else if (this->currentReference->nt == 2) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->currentFloating_gpu,
                                                    &this->currentFloating2_gpu, this->currentFloating->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the floating image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentFloating_gpu,
                                                           &this->currentFloating2_gpu, this->currentFloating)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the floating image");
            reg_exit();
        }
    }

    if (this->controlPointGrid_gpu != NULL) cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    if (cudaCommon_allocateArrayToDevice<float4>(&this->controlPointGrid_gpu, this->controlPointGrid->dim)) {
        reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
        reg_print_msg_error("Error when allocating the control point image");
        reg_exit();
    }
    if (cudaCommon_transferNiftiToArrayOnDevice<float4>(&this->controlPointGrid_gpu, this->controlPointGrid)) {
        reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
        reg_print_msg_error("Error when transferring the control point image");
        reg_exit();
    }

    int *targetMask_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&targetMask_h, this->activeVoxelNumber[this->currentLevel] * sizeof(int)));
    int *targetMask_h_ptr = &targetMask_h[0];
    for (int i = 0; i < this->currentReference->nx * this->currentReference->ny * this->currentReference->nz; i++) {
        if (this->currentMask[i] != -1)
            *targetMask_h_ptr++ = i;
    }
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->currentMask_gpu, this->activeVoxelNumber[this->currentLevel] * sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentMask_gpu, targetMask_h,
                                 this->activeVoxelNumber[this->currentLevel] * sizeof(int), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaFreeHost(targetMask_h));

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::InitialiseCurrentLevel");
#endif
    return maxStepSize;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearCurrentInputImage() {
    reg_f3d::ClearCurrentInputImage();

    if (cudaCommon_transferFromDeviceToNifti<float4>(this->controlPointGrid, &this->controlPointGrid_gpu)) {
        reg_print_fct_error("reg_f3d_gpu::ClearCurrentInputImage()");
        reg_print_msg_error("Error when transferring back the control point image");
        reg_exit();
    }
    cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    this->controlPointGrid_gpu = NULL;
    cudaCommon_free(&this->currentReference_gpu);
    this->currentReference_gpu = NULL;
    cudaCommon_free(&this->currentFloating_gpu);
    this->currentFloating_gpu = NULL;
    NR_CUDA_SAFE_CALL(cudaFree(this->currentMask_gpu));
    this->currentMask_gpu = NULL;

    if (this->currentReference2_gpu != NULL)
        cudaCommon_free(&this->currentReference2_gpu);
    this->currentReference2_gpu = NULL;
    if (this->currentFloating2_gpu != NULL)
        cudaCommon_free(&this->currentFloating2_gpu);
    this->currentFloating2_gpu = NULL;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::ClearCurrentInputImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::SetOptimiser() {
    if (this->useConjGradient)
        this->optimiser = new reg_conjugateGradient_gpu();
    else this->optimiser = new reg_optimiser_gpu();
    // The cpp and grad images are converted to float* instead of float4
    // to enable compatibility with the CPU class
    this->optimiser->Initialise(this->controlPointGrid->nvox,
                                this->controlPointGrid->nz > 1 ? 3 : 2,
                                this->optimiseX,
                                this->optimiseY,
                                this->optimiseZ,
                                this->maxiterationNumber,
                                0, // currentIterationNumber,
                                this,
                                reinterpret_cast<float*>(this->controlPointGrid_gpu),
                                reinterpret_cast<float*>(this->transformationGradient_gpu));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::SetOptimiser");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_f3d_gpu::NormaliseGradient() {
    // First compute the gradient max length for normalisation purpose
    float length = reg_getMaximalLength_gpu(&this->transformationGradient_gpu, this->optimiser->GetVoxNumber());

    if (strcmp(this->executableName, "NiftyReg F3D GPU") == 0) {
        // The gradient is normalised if we are running F3D
        // It will be normalised later when running symmetric or F3D2
#ifndef NDEBUG
        char text[255];
        sprintf(text, "Objective function gradient maximal length: %g", length);
        reg_print_msg_debug(text);
#endif
        reg_multiplyValue_gpu(this->optimiser->GetVoxNumber(), &this->transformationGradient_gpu, 1.f / length);
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::NormaliseGradient");
#endif
    // Returns the largest gradient distance
    return length;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
int reg_f3d_gpu::CheckMemoryMB() {
    if (!this->initialised)
        reg_f3d::Initialise();

    size_t referenceVoxelNumber = this->referencePyramid[this->levelToPerform - 1]->nx *
        this->referencePyramid[this->levelToPerform - 1]->ny *
        this->referencePyramid[this->levelToPerform - 1]->nz;

    size_t warpedVoxelNumber = this->referencePyramid[this->levelToPerform - 1]->nx *
        this->referencePyramid[this->levelToPerform - 1]->ny *
        this->referencePyramid[this->levelToPerform - 1]->nz *
        this->floatingPyramid[this->levelToPerform - 1]->nt;

    size_t totalMemoryRequiered = 0;
    // reference image
    totalMemoryRequiered += this->referencePyramid[this->levelToPerform - 1]->nvox * sizeof(float);

    // floating image
    totalMemoryRequiered += this->floatingPyramid[this->levelToPerform - 1]->nvox * sizeof(float);

    // warped image
    totalMemoryRequiered += warpedVoxelNumber * sizeof(float);

    // mask image
    totalMemoryRequiered += this->activeVoxelNumber[this->levelToPerform - 1] * sizeof(int);

    // deformation field
    totalMemoryRequiered += referenceVoxelNumber * sizeof(float4);

    // voxel based intensity gradient
    totalMemoryRequiered += referenceVoxelNumber * sizeof(float4);

    // voxel based NMI gradient + smoothing
    totalMemoryRequiered += 2 * referenceVoxelNumber * sizeof(float4);

    // control point grid
    size_t cp = 1;
    cp *= (int)floor(this->referencePyramid[this->levelToPerform - 1]->nx *
                     this->referencePyramid[this->levelToPerform - 1]->dx /
                     this->spacing[0]) + 5;
    cp *= (int)floor(this->referencePyramid[this->levelToPerform - 1]->ny *
                     this->referencePyramid[this->levelToPerform - 1]->dy /
                     this->spacing[1]) + 5;
    if (this->referencePyramid[this->levelToPerform - 1]->nz > 1)
        cp *= (int)floor(this->referencePyramid[this->levelToPerform - 1]->nz *
                         this->referencePyramid[this->levelToPerform - 1]->dz /
                         this->spacing[2]) + 5;
    totalMemoryRequiered += cp * sizeof(float4);

    // node based NMI gradient
    totalMemoryRequiered += cp * sizeof(float4);

    // conjugate gradient
    totalMemoryRequiered += 2 * cp * sizeof(float4);


    // HERE TODO

    // jacobian array
    if (this->jacobianLogWeight > 0)
        totalMemoryRequiered += 10 * referenceVoxelNumber * sizeof(float);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::CheckMemoryMB");
#endif
    return (int)(ceil((float)totalMemoryRequiered / float(1024 * 1024)));
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber) {
    if (this->measure_gpu_nmi == NULL)
        this->measure_gpu_nmi = new reg_nmi_gpu;
    this->measure_gpu_nmi->SetTimepointWeight(timepoint, 1.0);
    // I am here adding 4 to the specified bin number to accomodate for
    // the spline support
    this->measure_gpu_nmi->SetFloatingBinNumber(floBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseNMISetFloatingBinNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber) {
    if (this->measure_gpu_nmi == NULL)
        this->measure_gpu_nmi = new reg_nmi_gpu;
    this->measure_gpu_nmi->SetTimepointWeight(timepoint, 1.0);
    // I am here adding 4 to the specified bin number to accomodate for
    // the spline support
    this->measure_gpu_nmi->SetReferenceBinNumber(refBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseNMISetReferenceBinNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseSSD(int timepoint) {
    if (this->measure_gpu_ssd == NULL)
        this->measure_gpu_ssd = new reg_ssd_gpu;
    this->measure_gpu_ssd->SetTimepointWeight(timepoint, 1.0);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseSSD");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseKLDivergence(int timepoint) {
    if (this->measure_gpu_kld == NULL)
        this->measure_gpu_kld = new reg_kld_gpu;
    this->measure_gpu_kld->SetTimepointWeight(timepoint, 1.0);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseKLDivergence");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseLNCC(int timepoint, float stddev) {
    if (this->measure_gpu_lncc == NULL)
        this->measure_gpu_lncc = new reg_lncc_gpu;
    this->measure_gpu_lncc->SetTimepointWeight(timepoint, 1.0);
    this->measure_gpu_lncc->SetKernelStandardDeviation(timepoint, stddev);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseLNCC");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseDTI(int timepoint[6]) {
    reg_print_msg_error("The use of DTI has been deactivated as it requires some refactoring");
    reg_exit();

    // if(this->measure_gpu_dti==NULL)
    //    this->measure_gpu_dti=new reg_dti_gpu;
    // for(int i=0; i<6; ++i)
    //    this->measure_gpu_dti->SetActiveTimepoint(timepoint[i]);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::InitialiseSimilarity() {
    // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
    if (this->measure_gpu_nmi == NULL &&
        this->measure_gpu_ssd == NULL &&
        this->measure_gpu_dti == NULL &&
        this->measure_gpu_kld == NULL &&
        this->measure_gpu_lncc == NULL) {
        measure_gpu_nmi = new reg_nmi_gpu;
        for (int i = 0; i < this->inputReference->nt; ++i)
            measure_gpu_nmi->SetTimepointWeight(i, 1.0);
    }
    if (this->measure_gpu_nmi != NULL) {
        this->measure_gpu_nmi->InitialiseMeasure(this->currentReference,
                                                 this->currentFloating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warImgGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->currentReference_gpu,
                                                 &this->currentFloating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_nmi = this->measure_gpu_nmi;
    }

    if (this->measure_gpu_ssd != NULL) {
        this->measure_gpu_ssd->InitialiseMeasure(this->currentReference,
                                                 this->currentFloating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warImgGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 this->localWeightSimCurrent,
                                                 &this->currentReference_gpu,
                                                 &this->currentFloating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_ssd = this->measure_gpu_ssd;
    }

    if (this->measure_gpu_kld != NULL) {
        this->measure_gpu_kld->InitialiseMeasure(this->currentReference,
                                                 this->currentFloating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warImgGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->currentReference_gpu,
                                                 &this->currentFloating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_kld = this->measure_gpu_kld;
    }

    if (this->measure_gpu_lncc != NULL) {
        this->measure_gpu_lncc->InitialiseMeasure(this->currentReference,
                                                  this->currentFloating,
                                                  this->currentMask,
                                                  this->activeVoxelNumber[this->currentLevel],
                                                  this->warped,
                                                  this->warImgGradient,
                                                  this->voxelBasedMeasureGradient,
                                                  &this->currentReference_gpu,
                                                  &this->currentFloating_gpu,
                                                  &this->currentMask_gpu,
                                                  &this->warped_gpu,
                                                  &this->warpedGradientImage_gpu,
                                                  &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_lncc = this->measure_gpu_lncc;
    }

    if (this->measure_gpu_dti != NULL) {
        this->measure_gpu_dti->InitialiseMeasure(this->currentReference,
                                                 this->currentFloating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warImgGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->currentReference_gpu,
                                                 &this->currentFloating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_dti = this->measure_gpu_dti;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::InitialiseSimilarity()");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
