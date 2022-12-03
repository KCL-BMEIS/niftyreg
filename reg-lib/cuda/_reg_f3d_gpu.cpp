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

#include "_reg_f3d_gpu.h"

 /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
 /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::reg_f3d_gpu(int refTimePoint, int floTimePoint)
    : reg_f3d<float>::reg_f3d(refTimePoint, floTimePoint) {
    this->executableName = (char *)"NiftyReg F3D GPU";
    this->reference_gpu = nullptr;
    this->floating_gpu = nullptr;
    this->currentMask_gpu = nullptr;
    this->warped_gpu = nullptr;
    this->controlPointGrid_gpu = nullptr;
    this->deformationFieldImage_gpu = nullptr;
    this->warpedGradientImage_gpu = nullptr;
    this->voxelBasedMeasureGradientImage_gpu = nullptr;
    this->transformationGradient_gpu = nullptr;

    this->measure_gpu_ssd = nullptr;
    this->measure_gpu_kld = nullptr;
    this->measure_gpu_dti = nullptr;
    this->measure_gpu_lncc = nullptr;
    this->measure_gpu_nmi = nullptr;

    this->reference2_gpu = nullptr;
    this->floating2_gpu = nullptr;
    this->warped2_gpu = nullptr;
    this->warpedGradientImage2_gpu = nullptr;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::reg_f3d_gpu");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::~reg_f3d_gpu() {
    if (this->reference_gpu != nullptr)
        cudaCommon_free(&this->reference_gpu);
    if (this->floating_gpu != nullptr)
        cudaCommon_free(&this->floating_gpu);
    if (this->currentMask_gpu != nullptr)
        cudaCommon_free(&this->currentMask_gpu);
    if (this->warped_gpu != nullptr)
        cudaCommon_free(&this->warped_gpu);
    if (this->controlPointGrid_gpu != nullptr)
        cudaCommon_free(&this->controlPointGrid_gpu);
    if (this->deformationFieldImage_gpu != nullptr)
        cudaCommon_free(&this->deformationFieldImage_gpu);
    if (this->warpedGradientImage_gpu != nullptr)
        cudaCommon_free(&this->warpedGradientImage_gpu);
    if (this->voxelBasedMeasureGradientImage_gpu != nullptr)
        cudaCommon_free(&this->voxelBasedMeasureGradientImage_gpu);
    if (this->transformationGradient_gpu != nullptr)
        cudaCommon_free(&this->transformationGradient_gpu);

    if (this->reference2_gpu != nullptr)
        cudaCommon_free(&this->reference2_gpu);
    if (this->floating2_gpu != nullptr)
        cudaCommon_free(&this->floating2_gpu);
    if (this->warped2_gpu != nullptr)
        cudaCommon_free(&this->warped2_gpu);
    if (this->warpedGradientImage2_gpu != nullptr)
        cudaCommon_free(&this->warpedGradientImage2_gpu);

    if (this->optimiser != nullptr) {
        delete this->optimiser;
        this->optimiser = nullptr;
    }

    if (this->measure_gpu_nmi != nullptr) {
        delete this->measure_gpu_nmi;
        this->measure_gpu_nmi = nullptr;
        this->measure_nmi = nullptr;
    }
    if (this->measure_gpu_ssd != nullptr) {
        delete this->measure_gpu_ssd;
        this->measure_gpu_ssd = nullptr;
        this->measure_ssd = nullptr;
    }
    if (this->measure_gpu_kld != nullptr) {
        delete this->measure_gpu_kld;
        this->measure_gpu_kld = nullptr;
        this->measure_kld = nullptr;
    }
    if (this->measure_gpu_dti != nullptr) {
        delete this->measure_gpu_dti;
        this->measure_gpu_dti = nullptr;
        this->measure_dti = nullptr;
    }
    if (this->measure_gpu_lncc != nullptr) {
        delete this->measure_gpu_lncc;
        this->measure_gpu_lncc = nullptr;
        this->measure_lncc = nullptr;
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
void reg_f3d_gpu::DeallocateWarped() {
    reg_f3d::DeallocateWarped();

    if (this->warped_gpu != nullptr) {
        cudaCommon_free(&this->warped_gpu);
        this->warped_gpu = nullptr;
    }
    if (this->warped2_gpu != nullptr) {
        cudaCommon_free(&this->warped2_gpu);
        this->warped2_gpu = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateWarped");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateDeformationField() {
    this->DeallocateDeformationField();
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->deformationFieldImage_gpu,
                                 this->activeVoxelNumber[this->currentLevel] * sizeof(float4)));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::AllocateDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::DeallocateDeformationField() {
    if (this->deformationFieldImage_gpu != nullptr) {
        cudaCommon_free(&this->deformationFieldImage_gpu);
        this->deformationFieldImage_gpu = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateWarpedGradient() {
    this->DeallocateWarpedGradient();
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
void reg_f3d_gpu::DeallocateWarpedGradient() {
    if (this->warpedGradientImage_gpu != nullptr) {
        cudaCommon_free(&this->warpedGradientImage_gpu);
        this->warpedGradientImage_gpu = nullptr;
    }
    if (this->warpedGradientImage2_gpu != nullptr) {
        cudaCommon_free(&this->warpedGradientImage2_gpu);
        this->warpedGradientImage2_gpu = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateWarpedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateVoxelBasedMeasureGradient() {
    this->DeallocateVoxelBasedMeasureGradient();
    if (cudaCommon_allocateArrayToDevice(&this->voxelBasedMeasureGradientImage_gpu, this->reference->dim)) {
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
void reg_f3d_gpu::DeallocateVoxelBasedMeasureGradient() {
    if (this->voxelBasedMeasureGradientImage_gpu != nullptr) {
        cudaCommon_free(&this->voxelBasedMeasureGradientImage_gpu);
        this->voxelBasedMeasureGradientImage_gpu = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateVoxelBasedMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateTransformationGradient() {
    this->DeallocateTransformationGradient();
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
void reg_f3d_gpu::DeallocateTransformationGradient() {
    if (this->transformationGradient_gpu != nullptr) {
        cudaCommon_free(&this->transformationGradient_gpu);
        this->transformationGradient_gpu = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateTransformationGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeJacobianBasedPenaltyTerm(int type) {
    if (this->jacobianLogWeight <= 0) return 0;

    bool approx = type == 2 ? false : this->jacobianLogApproximation;

    double value = reg_spline_getJacobianPenaltyTerm_gpu(this->reference,
                                                         this->controlPointGrid,
                                                         this->controlPointGrid_gpu,
                                                         approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (value != value && it < maxit) {
        value = reg_spline_correctFolding_gpu(this->reference,
                                              this->controlPointGrid,
                                              this->controlPointGrid_gpu,
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

    double value = reg_spline_approxBendingEnergy_gpu(this->controlPointGrid,
                                                      this->controlPointGrid_gpu);
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
    if (this->controlPointGrid_gpu == nullptr) {
        reg_f3d::GetDeformationField();
    } else {
        // Compute the deformation field
        reg_spline_getDeformationField_gpu(this->controlPointGrid,
                                           this->reference,
                                           this->controlPointGrid_gpu,
                                           this->deformationFieldImage_gpu,
                                           this->currentMask_gpu,
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
    reg_resampleImage_gpu(this->floating,
                          this->warped_gpu,
                          this->floating_gpu,
                          this->deformationFieldImage_gpu,
                          this->currentMask_gpu,
                          this->activeVoxelNumber[this->currentLevel],
                          this->warpedPaddingValue);

    if (this->floating->nt == 2) {
        reg_resampleImage_gpu(this->floating,
                              this->warped2_gpu,
                              this->floating2_gpu,
                              this->deformationFieldImage_gpu,
                              this->currentMask_gpu,
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
               this->reference->nx * this->reference->ny * this->reference->nz *
               sizeof(float4));

    // The intensity gradient is first computed
    reg_getImageGradient_gpu(this->floating,
                             this->floating_gpu,
                             this->deformationFieldImage_gpu,
                             this->warpedGradientImage_gpu,
                             this->activeVoxelNumber[this->currentLevel],
                             this->warpedPaddingValue);

    // The gradient of the various measures of similarity are computed
    if (this->measure_gpu_nmi != nullptr)
        this->measure_gpu_nmi->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_ssd != nullptr)
        this->measure_gpu_ssd->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_kld != nullptr)
        this->measure_gpu_kld->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_lncc != nullptr)
        this->measure_gpu_lncc->GetVoxelBasedSimilarityMeasureGradient();

    if (this->measure_gpu_dti != nullptr)
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
        this->controlPointGrid->dx / this->reference->dx,
        this->controlPointGrid->dy / this->reference->dy,
        this->controlPointGrid->dz / this->reference->dz
    };
    reg_smoothImageForCubicSpline_gpu(this->warped,
                                      this->voxelBasedMeasureGradientImage_gpu,
                                      smoothingRadius);

    // The node gradient is extracted
    reg_voxelCentric2NodeCentric_gpu(this->warped,
                                     this->controlPointGrid,
                                     this->voxelBasedMeasureGradientImage_gpu,
                                     this->transformationGradient_gpu,
                                     this->similarityWeight);

    /* The similarity measure gradient is converted from voxel space to real space */
    mat44 *floatingMatrix_xyz = nullptr;
    if (this->floating->sform_code > 0)
        floatingMatrix_xyz = &(this->floating->sto_xyz);
    else floatingMatrix_xyz = &(this->floating->qto_xyz);
    reg_convertNMIGradientFromVoxelToRealSpace_gpu(floatingMatrix_xyz,
                                                   this->controlPointGrid,
                                                   this->transformationGradient_gpu);
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (this->gradientSmoothingSigma != 0) {
        reg_gaussianSmoothing_gpu(this->controlPointGrid,
                                  this->transformationGradient_gpu,
                                  this->gradientSmoothingSigma,
                                  nullptr);
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
                                               this->controlPointGrid_gpu,
                                               this->transformationGradient_gpu,
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

    reg_spline_getJacobianPenaltyTermGradient_gpu(this->reference,
                                                  this->controlPointGrid,
                                                  this->controlPointGrid_gpu,
                                                  this->transformationGradient_gpu,
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

    reg_updateControlPointPosition_gpu(this->controlPointGrid, currentDOF, bestDOF, gradient, scale);
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
nifti_image** reg_f3d_gpu::GetWarpedImage() {
    // The initial images are used
    if (this->inputReference == nullptr || this->inputFloating == nullptr || this->controlPointGrid == nullptr) {
        reg_print_fct_error("reg_f3d_gpu::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    this->reference = this->inputReference;
    this->floating = this->inputFloating;
    this->currentMask = (int*)calloc(this->activeVoxelNumber[this->currentLevel], sizeof(int));

    reg_tools_changeDatatype<float>(this->reference);
    reg_tools_changeDatatype<float>(this->floating);

    this->AllocateWarped();
    this->AllocateDeformationField();
    this->InitialiseCurrentLevel();
    this->WarpFloatingImage(3); // cubic spline interpolation
    this->DeallocateDeformationField();

    nifti_image **warpedImage = (nifti_image**)calloc(2, sizeof(nifti_image*));
    warpedImage[0] = nifti_copy_nim_info(this->warped);
    warpedImage[0]->cal_min = this->inputFloating->cal_min;
    warpedImage[0]->cal_max = this->inputFloating->cal_max;
    warpedImage[0]->scl_slope = this->inputFloating->scl_slope;
    warpedImage[0]->scl_inter = this->inputFloating->scl_inter;
    warpedImage[0]->data = (void*)malloc(warpedImage[0]->nvox * warpedImage[0]->nbyper);
    cudaCommon_transferFromDeviceToNifti(warpedImage[0], &this->warped_gpu);
    if (this->floating->nt == 2) {
        warpedImage[1] = warpedImage[0];
        warpedImage[1]->data = (void*)malloc(warpedImage[1]->nvox * warpedImage[1]->nbyper);
        cudaCommon_transferFromDeviceToNifti(warpedImage[1], &this->warped2_gpu);
    }

    this->DeallocateWarped();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::GetWarpedImage");
#endif
    return warpedImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_f3d_gpu::InitialiseCurrentLevel() {
    float maxStepSize = reg_f3d::InitialiseCurrentLevel();

    if (this->reference_gpu != nullptr) cudaCommon_free(&this->reference_gpu);
    if (this->reference2_gpu != nullptr) cudaCommon_free(&this->reference2_gpu);
    if (this->reference->nt == 1) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->reference_gpu, this->reference->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the reference image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->reference_gpu, this->reference)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the reference image");
            reg_exit();
        }
    } else if (this->reference->nt == 2) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->reference_gpu,
                                                    &this->reference2_gpu, this->reference->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the reference image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->reference_gpu,
                                                           &this->reference2_gpu, this->reference)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the reference image");
            reg_exit();
        }
    }

    if (this->floating_gpu != nullptr) cudaCommon_free(&this->floating_gpu);
    if (this->floating2_gpu != nullptr) cudaCommon_free(&this->floating2_gpu);
    if (this->reference->nt == 1) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->floating_gpu, this->floating->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the floating image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->floating_gpu, this->floating)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the floating image");
            reg_exit();
        }
    } else if (this->reference->nt == 2) {
        if (cudaCommon_allocateArrayToDevice<float>(&this->floating_gpu,
                                                    &this->floating2_gpu, this->floating->dim)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when allocating the floating image");
            reg_exit();
        }
        if (cudaCommon_transferNiftiToArrayOnDevice<float>(&this->floating_gpu,
                                                           &this->floating2_gpu, this->floating)) {
            reg_print_fct_error("reg_f3d_gpu::InitialiseCurrentLevel()");
            reg_print_msg_error("Error when transferring the floating image");
            reg_exit();
        }
    }

    if (this->controlPointGrid_gpu != nullptr) cudaCommon_free(&this->controlPointGrid_gpu);
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
    for (int i = 0; i < this->reference->nx * this->reference->ny * this->reference->nz; i++) {
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
void reg_f3d_gpu::DeallocateCurrentInputImage() {
    reg_f3d::DeallocateCurrentInputImage();

    if (cudaCommon_transferFromDeviceToNifti<float4>(this->controlPointGrid, &this->controlPointGrid_gpu)) {
        reg_print_fct_error("reg_f3d_gpu::DeallocateCurrentInputImage()");
        reg_print_msg_error("Error when transferring back the control point image");
        reg_exit();
    }
    cudaCommon_free(&this->controlPointGrid_gpu);
    this->controlPointGrid_gpu = nullptr;
    cudaCommon_free(&this->reference_gpu);
    this->reference_gpu = nullptr;
    cudaCommon_free(&this->floating_gpu);
    this->floating_gpu = nullptr;
    NR_CUDA_SAFE_CALL(cudaFree(this->currentMask_gpu));
    this->currentMask_gpu = nullptr;

    if (this->reference2_gpu != nullptr)
        cudaCommon_free(&this->reference2_gpu);
    this->reference2_gpu = nullptr;
    if (this->floating2_gpu != nullptr)
        cudaCommon_free(&this->floating2_gpu);
    this->floating2_gpu = nullptr;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::DeallocateCurrentInputImage");
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
                                this->maxIterationNumber,
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
    float length = reg_getMaximalLength_gpu(this->transformationGradient_gpu, this->optimiser->GetVoxNumber());

    if (strcmp(this->executableName, "NiftyReg F3D GPU") == 0) {
        // The gradient is normalised if we are running F3D
        // It will be normalised later when running symmetric or F3D2
#ifndef NDEBUG
        char text[255];
        sprintf(text, "Objective function gradient maximal length: %g", length);
        reg_print_msg_debug(text);
#endif
        reg_multiplyValue_gpu(this->optimiser->GetVoxNumber(), this->transformationGradient_gpu, 1.f / length);
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
    if (this->measure_gpu_nmi == nullptr)
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
    if (this->measure_gpu_nmi == nullptr)
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
    if (this->measure_gpu_ssd == nullptr)
        this->measure_gpu_ssd = new reg_ssd_gpu;
    this->measure_gpu_ssd->SetTimepointWeight(timepoint, 1.0);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseSSD");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseKLDivergence(int timepoint) {
    if (this->measure_gpu_kld == nullptr)
        this->measure_gpu_kld = new reg_kld_gpu;
    this->measure_gpu_kld->SetTimepointWeight(timepoint, 1.0);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d_gpu::UseKLDivergence");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseLNCC(int timepoint, float stddev) {
    if (this->measure_gpu_lncc == nullptr)
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

    // if(this->measure_gpu_dti==nullptr)
    //    this->measure_gpu_dti=new reg_dti_gpu;
    // for(int i=0; i<6; ++i)
    //    this->measure_gpu_dti->SetActiveTimepoint(timepoint[i]);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::InitialiseSimilarity() {
    // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
    if (this->measure_gpu_nmi == nullptr &&
        this->measure_gpu_ssd == nullptr &&
        this->measure_gpu_dti == nullptr &&
        this->measure_gpu_kld == nullptr &&
        this->measure_gpu_lncc == nullptr) {
        measure_gpu_nmi = new reg_nmi_gpu;
        for (int i = 0; i < this->inputReference->nt; ++i)
            measure_gpu_nmi->SetTimepointWeight(i, 1.0);
    }
    if (this->measure_gpu_nmi != nullptr) {
        this->measure_gpu_nmi->InitialiseMeasure(this->reference,
                                                 this->floating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warpedGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->reference_gpu,
                                                 &this->floating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_nmi = this->measure_gpu_nmi;
    }

    if (this->measure_gpu_ssd != nullptr) {
        this->measure_gpu_ssd->InitialiseMeasure(this->reference,
                                                 this->floating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warpedGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 this->localWeightSimCurrent,
                                                 &this->reference_gpu,
                                                 &this->floating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_ssd = this->measure_gpu_ssd;
    }

    if (this->measure_gpu_kld != nullptr) {
        this->measure_gpu_kld->InitialiseMeasure(this->reference,
                                                 this->floating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warpedGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->reference_gpu,
                                                 &this->floating_gpu,
                                                 &this->currentMask_gpu,
                                                 &this->warped_gpu,
                                                 &this->warpedGradientImage_gpu,
                                                 &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_kld = this->measure_gpu_kld;
    }

    if (this->measure_gpu_lncc != nullptr) {
        this->measure_gpu_lncc->InitialiseMeasure(this->reference,
                                                  this->floating,
                                                  this->currentMask,
                                                  this->activeVoxelNumber[this->currentLevel],
                                                  this->warped,
                                                  this->warpedGradient,
                                                  this->voxelBasedMeasureGradient,
                                                  &this->reference_gpu,
                                                  &this->floating_gpu,
                                                  &this->currentMask_gpu,
                                                  &this->warped_gpu,
                                                  &this->warpedGradientImage_gpu,
                                                  &this->voxelBasedMeasureGradientImage_gpu);
        this->measure_lncc = this->measure_gpu_lncc;
    }

    if (this->measure_gpu_dti != nullptr) {
        this->measure_gpu_dti->InitialiseMeasure(this->reference,
                                                 this->floating,
                                                 this->currentMask,
                                                 this->activeVoxelNumber[this->currentLevel],
                                                 this->warped,
                                                 this->warpedGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 &this->reference_gpu,
                                                 &this->floating_gpu,
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
