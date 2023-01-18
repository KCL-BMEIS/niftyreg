/*
 *  _reg_f3d2.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d2.h"

/* *************************************************************** */
template <class T>
reg_f3d2<T>::reg_f3d2(int refTimePoint, int floTimePoint):
    reg_f3d<T>::reg_f3d(refTimePoint, floTimePoint) {
    this->executableName = (char*)"NiftyReg F3D2";
    backwardControlPointGrid = nullptr;
    backwardWarped = nullptr;
    backwardWarpedGradientImage = nullptr;
    backwardDeformationFieldImage = nullptr;
    backwardVoxelBasedMeasureGradientImage = nullptr;
    backwardTransformationGradient = nullptr;
    floatingMaskImage = nullptr;
    floatingMask = nullptr;
    floatingMaskPyramid = nullptr;
    backwardActiveVoxelNumber = nullptr;
    backwardJacobianMatrix = nullptr;
    inverseConsistencyWeight = 0;
    bchUpdate = false;
    useGradientCumulativeExp = true;
    bchUpdateValue = 0;

#ifndef NDEBUG
    reg_print_msg_debug("reg_f3d2 constructor called");
#endif
}
/* *************************************************************** */
template <class T>
reg_f3d2<T>::~reg_f3d2() {
    if (backwardControlPointGrid) {
        nifti_image_free(backwardControlPointGrid);
        backwardControlPointGrid = nullptr;
    }

    if (floatingMaskPyramid) {
        if (this->usePyramid) {
            for (unsigned int i = 0; i < this->levelToPerform; i++) {
                if (floatingMaskPyramid[i]) {
                    free(floatingMaskPyramid[i]);
                    floatingMaskPyramid[i] = nullptr;
                }
            }
        } else {
            if (floatingMaskPyramid[0]) {
                free(floatingMaskPyramid[0]);
                floatingMaskPyramid[0] = nullptr;
            }
        }
        free(floatingMaskPyramid);
        floatingMaskPyramid = nullptr;
    }

    if (backwardActiveVoxelNumber) {
        free(backwardActiveVoxelNumber);
        backwardActiveVoxelNumber = nullptr;
    }
#ifndef NDEBUG
    reg_print_msg_debug("reg_f3d2 destructor called");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::SetFloatingMask(nifti_image *m) {
    floatingMaskImage = m;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::~SetFloatingMask");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::SetInverseConsistencyWeight(T w) {
    inverseConsistencyWeight = w;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::SetInverseConsistencyWeight");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::InitialiseCurrentLevel() {
    // Refine the control point grids if required
    if (this->gridRefinement) {
        if (this->currentLevel == 0) {
            this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16, this->levelNumber - 1));
            this->linearEnergyWeight = this->linearEnergyWeight / static_cast<T>(powf(3, this->levelNumber - 1));
        } else {
            reg_spline_refineControlPointGrid(this->controlPointGrid);
            reg_spline_refineControlPointGrid(backwardControlPointGrid);
            this->bendingEnergyWeight = this->bendingEnergyWeight * static_cast<T>(16);
            this->linearEnergyWeight = this->linearEnergyWeight * static_cast<T>(3);
        }
    }

    // Set the mask images
    if (this->usePyramid) {
        this->currentMask = this->maskPyramid[this->currentLevel];
        floatingMask = floatingMaskPyramid[this->currentLevel];
    } else {
        this->currentMask = this->maskPyramid[0];
        floatingMask = floatingMaskPyramid[0];
    }

    // Define the initial step size for the gradient ascent optimisation
    T maxStepSize = this->reference->dx;
    maxStepSize = this->reference->dy > maxStepSize ? this->reference->dy : maxStepSize;
    maxStepSize = this->floating->dx > maxStepSize ? this->floating->dx : maxStepSize;
    maxStepSize = this->floating->dy > maxStepSize ? this->floating->dy : maxStepSize;
    if (this->reference->ndim > 2) {
        maxStepSize = (this->reference->dz > maxStepSize) ? this->reference->dz : maxStepSize;
        maxStepSize = (this->floating->dz > maxStepSize) ? this->floating->dz : maxStepSize;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::InitialiseCurrentLevel");
#endif
    return maxStepSize;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateCurrentInputImage() {
    reg_f3d<T>::DeallocateCurrentInputImage();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateCurrentInputImage");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::AllocateWarped() {
    DeallocateWarped();

    reg_f3d<T>::AllocateWarped();
    if (!this->floating) {
        reg_print_fct_error("reg_f3d2<T>::AllocateWarped()");
        reg_print_msg_error("The floating image is not defined");
        reg_exit();
    }
    backwardWarped = nifti_copy_nim_info(this->floating);
    backwardWarped->dim[0] = backwardWarped->ndim = this->reference->ndim;
    backwardWarped->dim[4] = backwardWarped->nt = this->reference->nt;
    backwardWarped->pixdim[4] = backwardWarped->dt = 1;
    backwardWarped->nvox = size_t(backwardWarped->nx * backwardWarped->ny * backwardWarped->nz * backwardWarped->nt);
    backwardWarped->datatype = this->reference->datatype;
    backwardWarped->nbyper = this->reference->nbyper;
    backwardWarped->data = calloc(backwardWarped->nvox, backwardWarped->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::AllocateWarped");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateWarped() {
    reg_f3d<T>::DeallocateWarped();
    if (backwardWarped) {
        nifti_image_free(backwardWarped);
        backwardWarped = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateWarped");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::AllocateDeformationField() {
    DeallocateDeformationField();

    reg_f3d<T>::AllocateDeformationField();
    if (!this->floating) {
        reg_print_fct_error("reg_f3d2<T>::AllocateDeformationField()");
        reg_print_msg_error("The floating image is not defined");
        reg_exit();
    }
    if (!backwardControlPointGrid) {
        reg_print_fct_error("reg_f3d2<T>::AllocateDeformationField()");
        reg_print_msg_error("The backward control point image is not defined");
        reg_exit();
    }
    backwardDeformationFieldImage = nifti_copy_nim_info(this->floating);
    backwardDeformationFieldImage->dim[0] = backwardDeformationFieldImage->ndim = 5;
    backwardDeformationFieldImage->dim[1] = backwardDeformationFieldImage->nx = this->floating->nx;
    backwardDeformationFieldImage->dim[2] = backwardDeformationFieldImage->ny = this->floating->ny;
    backwardDeformationFieldImage->dim[3] = backwardDeformationFieldImage->nz = this->floating->nz;
    backwardDeformationFieldImage->dim[4] = backwardDeformationFieldImage->nt = 1;
    backwardDeformationFieldImage->pixdim[4] = backwardDeformationFieldImage->dt = 1;
    if (this->floating->nz == 1)
        backwardDeformationFieldImage->dim[5] = backwardDeformationFieldImage->nu = 2;
    else backwardDeformationFieldImage->dim[5] = backwardDeformationFieldImage->nu = 3;
    backwardDeformationFieldImage->pixdim[5] = backwardDeformationFieldImage->du = 1;
    backwardDeformationFieldImage->dim[6] = backwardDeformationFieldImage->nv = 1;
    backwardDeformationFieldImage->pixdim[6] = backwardDeformationFieldImage->dv = 1;
    backwardDeformationFieldImage->dim[7] = backwardDeformationFieldImage->nw = 1;
    backwardDeformationFieldImage->pixdim[7] = backwardDeformationFieldImage->dw = 1;
    backwardDeformationFieldImage->nvox = size_t(backwardDeformationFieldImage->nx * backwardDeformationFieldImage->ny *
                                                 backwardDeformationFieldImage->nz * backwardDeformationFieldImage->nt *
                                                 backwardDeformationFieldImage->nu);
    backwardDeformationFieldImage->nbyper = backwardControlPointGrid->nbyper;
    backwardDeformationFieldImage->datatype = backwardControlPointGrid->datatype;
    backwardDeformationFieldImage->data = calloc(backwardDeformationFieldImage->nvox,
                                                 backwardDeformationFieldImage->nbyper);
    backwardDeformationFieldImage->intent_code = NIFTI_INTENT_VECTOR;
    memset(backwardDeformationFieldImage->intent_name, 0, 16);
    strcpy(backwardDeformationFieldImage->intent_name, "NREG_TRANS");
    backwardDeformationFieldImage->intent_p1 = DEF_FIELD;
    backwardDeformationFieldImage->scl_slope = 1;
    backwardDeformationFieldImage->scl_inter = 0;

    if (this->measure_dti)
        backwardJacobianMatrix = (mat33*)malloc(backwardDeformationFieldImage->nx * backwardDeformationFieldImage->ny *
                                                backwardDeformationFieldImage->nz * sizeof(mat33));

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::AllocateDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateDeformationField() {
    reg_f3d<T>::DeallocateDeformationField();
    if (backwardDeformationFieldImage) {
        nifti_image_free(backwardDeformationFieldImage);
        backwardDeformationFieldImage = nullptr;
    }
    if (backwardJacobianMatrix) {
        free(backwardJacobianMatrix);
        backwardJacobianMatrix = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::AllocateWarpedGradient() {
    DeallocateWarpedGradient();

    reg_f3d<T>::AllocateWarpedGradient();
    if (!backwardDeformationFieldImage) {
        reg_print_fct_error("reg_f3d2<T>::AllocateWarpedGradient()");
        reg_print_msg_error("The backward control point image is not defined");
        reg_exit();
    }
    backwardWarpedGradientImage = nifti_copy_nim_info(backwardDeformationFieldImage);
    backwardWarpedGradientImage->data = calloc(backwardWarpedGradientImage->nvox,
                                               backwardWarpedGradientImage->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::AllocateWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateWarpedGradient() {
    reg_f3d<T>::DeallocateWarpedGradient();
    if (backwardWarpedGradientImage) {
        nifti_image_free(backwardWarpedGradientImage);
        backwardWarpedGradientImage = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::AllocateVoxelBasedMeasureGradient() {
    DeallocateVoxelBasedMeasureGradient();

    reg_f3d<T>::AllocateVoxelBasedMeasureGradient();
    if (!backwardDeformationFieldImage) {
        reg_print_fct_error("reg_f3d2<T>::AllocateVoxelBasedMeasureGradient()");
        reg_print_msg_error("The backward control point image is not defined");
        reg_exit();
    }
    backwardVoxelBasedMeasureGradientImage = nifti_copy_nim_info(backwardDeformationFieldImage);
    backwardVoxelBasedMeasureGradientImage->data = calloc(backwardVoxelBasedMeasureGradientImage->nvox,
                                                          backwardVoxelBasedMeasureGradientImage->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::AllocateVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateVoxelBasedMeasureGradient() {
    reg_f3d<T>::DeallocateVoxelBasedMeasureGradient();
    if (backwardVoxelBasedMeasureGradientImage) {
        nifti_image_free(backwardVoxelBasedMeasureGradientImage);
        backwardVoxelBasedMeasureGradientImage = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::AllocateTransformationGradient() {
    DeallocateTransformationGradient();

    reg_f3d<T>::AllocateTransformationGradient();
    if (!backwardControlPointGrid) {
        reg_print_fct_error("reg_f3d2<T>::AllocateTransformationGradient()");
        reg_print_msg_error("The backward control point image is not defined");
        reg_exit();
    }
    backwardTransformationGradient = nifti_copy_nim_info(backwardControlPointGrid);
    backwardTransformationGradient->data = calloc(backwardTransformationGradient->nvox,
                                                  backwardTransformationGradient->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::AllocateTransformationGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DeallocateTransformationGradient() {
    reg_f3d<T>::DeallocateTransformationGradient();
    if (backwardTransformationGradient) {
        nifti_image_free(backwardTransformationGradient);
        backwardTransformationGradient = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DeallocateTransformationGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::CheckParameters() {
    reg_f3d<T>::CheckParameters();

    // CHECK THE FLOATING MASK DIMENSION IF IT IS DEFINED
    if (floatingMaskImage) {
        if (this->inputFloating->nx != floatingMaskImage->nx ||
            this->inputFloating->ny != floatingMaskImage->ny ||
            this->inputFloating->nz != floatingMaskImage->nz) {
            reg_print_fct_error("reg_f3d2<T>::CheckParameters()");
            reg_print_msg_error("The floating image and its mask have different dimension");
            reg_exit();
        }
    }

    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    T penaltySum = (this->bendingEnergyWeight + this->linearEnergyWeight + this->jacobianLogWeight +
                    inverseConsistencyWeight + this->landmarkRegWeight);
    if (penaltySum >= 1) {
        this->similarityWeight = 0;
        this->bendingEnergyWeight /= penaltySum;
        this->linearEnergyWeight /= penaltySum;
        this->jacobianLogWeight /= penaltySum;
        inverseConsistencyWeight /= penaltySum;
        this->landmarkRegWeight /= penaltySum;
    } else this->similarityWeight = 1 - penaltySum;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::CheckParameters");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetDeformationField() {
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   false, //composition
                                   true); // bspline
    reg_spline_getDeformationField(backwardControlPointGrid,
                                   backwardDeformationFieldImage,
                                   floatingMask,
                                   false, //composition
                                   true); // bspline

    // By default the number of steps is automatically updated
    bool updateStepNumber = true;
    // The provided step number is used for the final resampling
    if (!this->optimiser)
        updateStepNumber = false;
#ifndef NDEBUG
    char text[255];
    sprintf(text, "Velocity integration forward. Step number update=%i", updateStepNumber);
    reg_print_msg_debug(text);
#endif
    // The forward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDefFieldFromVelocityGrid(this->controlPointGrid,
                                           this->deformationFieldImage,
                                           updateStepNumber);
#ifndef NDEBUG
    sprintf(text, "Velocity integration backward. Step number update=%i", updateStepNumber);
    reg_print_msg_debug(text);
#endif
    // The number of step number is copied over from the forward transformation
    backwardControlPointGrid->intent_p2 = this->controlPointGrid->intent_p2;
    // The backward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDefFieldFromVelocityGrid(backwardControlPointGrid,
                                           backwardDeformationFieldImage,
                                           false);
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::WarpFloatingImage(int inter) {
    // Compute the deformation fields
    GetDeformationField();

    // Resample the floating image
    if (!this->measure_dti) {
        reg_resampleImage(this->floating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          inter,
                          this->warpedPaddingValue);
    } else {
        reg_defField_getJacobianMatrix(this->deformationFieldImage,
                                       this->forwardJacobianMatrix);
        /*DTI needs fixing!
        reg_resampleImage(this->floating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          inter,
                          this->warpedPaddingValue,
                          this->measure_dti->GetActiveTimepoints(),
                          this->forwardJacobianMatrix);*/
    }

    // Resample the reference image
    if (!this->measure_dti) {
        reg_resampleImage(this->reference, // input image
                          backwardWarped, // warped input image
                          backwardDeformationFieldImage, // deformation field
                          floatingMask, // mask
                          inter, // interpolation type
                          this->warpedPaddingValue); // padding value
    } else {
        reg_defField_getJacobianMatrix(backwardDeformationFieldImage,
                                       backwardJacobianMatrix);
        /* DTI needs fixing
       reg_resampleImage(this->reference, // input image
                           backwardWarped, // warped input image
                           backwardDeformationFieldImage, // deformation field
                           floatingMask, // mask
                           inter, // interpolation type
                           this->warpedPaddingValue, // padding value
                           this->measure_dti->GetActiveTimepoints(),
                           backwardJacobianMatrix);*/
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::WarpFloatingImage");
#endif
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeJacobianBasedPenaltyTerm(int type) {
    if (this->jacobianLogWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(type);

    bool approx = type == 2 ? false : this->jacobianLogApproximation;

    double backwardPenaltyTerm = reg_spline_getJacobianPenaltyTerm(backwardControlPointGrid,
                                                                   this->floating,
                                                                   approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (backwardPenaltyTerm != backwardPenaltyTerm && it < maxit) {
        backwardPenaltyTerm = reg_spline_correctFolding(backwardControlPointGrid,
                                                        this->floating,
                                                        approx);
#ifndef NDEBUG
        reg_print_msg_debug("Folding correction - Backward transformation");
#endif
        it++;
    }
    if (type > 0 && it > 0) {
        if (backwardPenaltyTerm != backwardPenaltyTerm) {
            this->optimiser->RestoreBestDOF();
#ifndef NDEBUG
            reg_print_fct_warn("reg_f3d2<T>::ComputeJacobianBasedPenaltyTerm()");
            reg_print_msg_warn("The backward transformation folding correction scheme failed");
#endif
        } else {
#ifdef NDEBUG
            if (this->verbose) {
#endif
                char text[255];
                sprintf(text, "Backward transformation folding correction, %i step(s)", it);
                reg_print_msg_debug(text);
#ifdef NDEBUG
            }
#endif
        }
    }
    backwardPenaltyTerm *= this->jacobianLogWeight;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ComputeJacobianBasedPenaltyTerm");
#endif
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeBendingEnergyPenaltyTerm() {
    if (this->bendingEnergyWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();

    double value = reg_spline_approxBendingEnergy(backwardControlPointGrid);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ComputeBendingEnergyPenaltyTerm");
#endif
    return forwardPenaltyTerm + this->bendingEnergyWeight * value;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeLinearEnergyPenaltyTerm() {
    if (this->linearEnergyWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();

    double backwardPenaltyTerm = this->linearEnergyWeight * reg_spline_approxLinearEnergy(backwardControlPointGrid);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ComputeLinearEnergyPenaltyTerm");
#endif
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeLandmarkDistancePenaltyTerm() {
    if (this->landmarkRegWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm();

    double backwardPenaltyTerm = this->landmarkRegWeight * reg_spline_getLandmarkDistance(backwardControlPointGrid,
                                                                                          this->landmarkRegNumber,
                                                                                          this->landmarkFloating,
                                                                                          this->landmarkReference);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ComputeLandmarkDistancePenaltyTerm");
#endif
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient() {
    // The voxel based gradient image is initialised with zeros
    reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradient,
                                   this->voxelBasedMeasureGradient,
                                   0);
    reg_tools_multiplyValueToImage(backwardVoxelBasedMeasureGradientImage,
                                   backwardVoxelBasedMeasureGradientImage,
                                   0);
    // The intensity gradient is first computed
    //    if(this->measure_dti!=nullptr){
    //        reg_getImageGradient(this->floating,
    //                             this->warpedGradient,
    //                             this->deformationFieldImage,
    //                             this->currentMask,
    //                             this->interpolation,
    //                             this->warpedPaddingValue,
    //                             this->measure_dti->GetActiveTimepoints(),
    //                             this->forwardJacobianMatrix,
    //                             this->warped);

    //        reg_getImageGradient(this->reference,
    //                             backwardWarpedGradientImage,
    //                             backwardDeformationFieldImage,
    //                             floatingMask,
    //                             this->interpolation,
    //                             this->warpedPaddingValue,
    //                             this->measure_dti->GetActiveTimepoints(),
    //                             backwardJacobianMatrix,
    //                             backwardWarped);
    //   if(this->measure_dti!=nullptr)
    //      this->measure_dti->GetVoxelBasedSimilarityMeasureGradient();
    //    }
    //    else{
    //    }


    for (int t = 0; t < this->reference->nt; ++t) {
        reg_getImageGradient(this->floating,
                             this->warpedGradient,
                             this->deformationFieldImage,
                             this->currentMask,
                             this->interpolation,
                             this->warpedPaddingValue,
                             t);

        reg_getImageGradient(this->reference,
                             backwardWarpedGradientImage,
                             backwardDeformationFieldImage,
                             floatingMask,
                             this->interpolation,
                             this->warpedPaddingValue,
                             t);

        // The gradient of the various measures of similarity are computed
        if (this->measure_nmi)
            this->measure_nmi->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_ssd)
            this->measure_ssd->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_kld)
            this->measure_kld->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_lncc)
            this->measure_lncc->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_mind)
            this->measure_mind->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_mindssc)
            this->measure_mindssc->GetVoxelBasedSimilarityMeasureGradient(t);
    } // timepoint

    // Exponentiate the gradients if required
    ExponentiateGradient();

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetVoxelBasedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetSimilarityMeasureGradient() {
    reg_f3d<T>::GetSimilarityMeasureGradient();

    // The voxel based sim measure gradient is convolved with a spline kernel
    // Convolution along the x axis
    float currentNodeSpacing[3];
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dx;
    bool activeAxis[3] = {1, 0, 0};
    reg_tools_kernelConvolution(backwardVoxelBasedMeasureGradientImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // mask
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dy;
    activeAxis[0] = 0;
    activeAxis[1] = 1;
    reg_tools_kernelConvolution(backwardVoxelBasedMeasureGradientImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // mask
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the z axis if required
    if (this->voxelBasedMeasureGradient->nz > 1) {
        currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dz;
        activeAxis[1] = 0;
        activeAxis[2] = 1;
        reg_tools_kernelConvolution(backwardVoxelBasedMeasureGradientImage,
                                    currentNodeSpacing,
                                    CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                    nullptr, // mask
                                    nullptr, // all volumes are active
                                    activeAxis);
    }
    // The backward node based sim measure gradient is extracted
    mat44 reorientation;
    if (this->reference->sform_code > 0)
        reorientation = this->reference->sto_ijk;
    else reorientation = this->reference->qto_ijk;
    reg_voxelCentric2NodeCentric(backwardTransformationGradient,
                                 backwardVoxelBasedMeasureGradientImage,
                                 this->similarityWeight,
                                 false, // no update
                                 &reorientation); // voxel to mm conversion
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetSimilarityMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetJacobianBasedGradient() {
    if (this->jacobianLogWeight <= 0) return;

    reg_f3d<T>::GetJacobianBasedGradient();

    reg_spline_getJacobianPenaltyTermGradient(backwardControlPointGrid,
                                              this->floating,
                                              backwardTransformationGradient,
                                              this->jacobianLogWeight,
                                              this->jacobianLogApproximation);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetBendingEnergyGradient() {
    if (this->bendingEnergyWeight <= 0) return;

    reg_f3d<T>::GetBendingEnergyGradient();

    reg_spline_approxBendingEnergyGradient(backwardControlPointGrid,
                                           backwardTransformationGradient,
                                           this->bendingEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetBendingEnergyGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetLinearEnergyGradient() {
    if (this->linearEnergyWeight <= 0) return;

    reg_f3d<T>::GetLinearEnergyGradient();

    reg_spline_approxLinearEnergyGradient(backwardControlPointGrid,
                                          backwardTransformationGradient,
                                          this->linearEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetLinearEnergyGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetLandmarkDistanceGradient() {
    if (this->landmarkRegWeight <= 0) return;

    reg_f3d<T>::GetLandmarkDistanceGradient();

    reg_spline_getLandmarkDistanceGradient(backwardControlPointGrid,
                                           backwardTransformationGradient,
                                           this->landmarkRegNumber,
                                           this->landmarkFloating,
                                           this->landmarkReference,
                                           this->landmarkRegWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetLandmarkDistanceGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SetGradientImageToZero() {
    reg_f3d<T>::SetGradientImageToZero();

    T *nodeGradPtr = static_cast<T*>(backwardTransformationGradient->data);
    for (size_t i = 0; i < backwardTransformationGradient->nvox; ++i)
        *nodeGradPtr++ = 0;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::SetGradientImageToZero");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SmoothGradient() {
    if (this->gradientSmoothingSigma != 0) {
        reg_f3d<T>::SmoothGradient();
        // The gradient is smoothed using a Gaussian kernel if it is required
        float kernel = fabs(this->gradientSmoothingSigma);
        reg_tools_kernelConvolution(backwardTransformationGradient,
                                    &kernel,
                                    GAUSSIAN_KERNEL);
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetApproximatedGradient() {
    reg_f3d<T>::GetApproximatedGradient();

    // Loop over every control points
    T *gridPtr = static_cast<T*>(backwardControlPointGrid->data);
    T *gradPtr = static_cast<T*>(backwardTransformationGradient->data);
    T eps = this->floating->dx / 1000.f;
    for (size_t i = 0; i < backwardControlPointGrid->nvox; i++) {
        T currentValue = this->optimiser->GetBestDOF_b()[i];
        gridPtr[i] = currentValue + eps;
        double valPlus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue - eps;
        double valMinus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue;
        gradPtr[i] = -(T)((valPlus - valMinus) / (2.0 * eps));
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::NormaliseGradient() {
    // The forward gradient max length is computed
    T forwardMaxValue = reg_f3d<T>::NormaliseGradient();

    // The backward gradient max length is computed
    T maxGradValue = 0;
    size_t voxNumber = backwardTransformationGradient->nx * backwardTransformationGradient->ny * backwardTransformationGradient->nz;
    T *bckPtrX = static_cast<T*>(backwardTransformationGradient->data);
    T *bckPtrY = &bckPtrX[voxNumber];
    if (backwardTransformationGradient->nz > 1) {
        T *bckPtrZ = &bckPtrY[voxNumber];
        for (size_t i = 0; i < voxNumber; i++) {
            T valX = 0, valY = 0, valZ = 0;
            if (this->optimiseX)
                valX = *bckPtrX++;
            if (this->optimiseY)
                valY = *bckPtrY++;
            if (this->optimiseZ)
                valZ = *bckPtrZ++;
            T length = (T)(sqrt(valX * valX + valY * valY + valZ * valZ));
            maxGradValue = (length > maxGradValue) ? length : maxGradValue;
        }
    } else {
        for (size_t i = 0; i < voxNumber; i++) {
            T valX = 0, valY = 0;
            if (this->optimiseX)
                valX = *bckPtrX++;
            if (this->optimiseY)
                valY = *bckPtrY++;
            T length = (T)(sqrt(valX * valX + valY * valY));
            maxGradValue = (length > maxGradValue) ? length : maxGradValue;
        }
    }

    // The largest value between the forward and backward gradient is kept
    maxGradValue = maxGradValue > forwardMaxValue ? maxGradValue : forwardMaxValue;
#ifndef NDEBUG
    char text[255];
    sprintf(text, "Objective function gradient maximal length: %g", maxGradValue);
    reg_print_msg_debug(text);
#endif

    // The forward gradient is normalised
    T *forPtrX = static_cast<T*>(this->transformationGradient->data);
    for (size_t i = 0; i < this->transformationGradient->nvox; ++i) {
        *forPtrX++ /= maxGradValue;
    }
    // The backward gradient is normalised
    bckPtrX = static_cast<T*>(backwardTransformationGradient->data);
    for (size_t i = 0; i < backwardTransformationGradient->nvox; ++i) {
        *bckPtrX++ /= maxGradValue;
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::NormaliseGradient");
#endif
    // Returns the largest gradient distance
    return maxGradValue;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::GetObjectiveFunctionGradient() {
    if (!this->useApproxGradient) {
        // Compute the gradient of the similarity measure
        if (this->similarityWeight > 0) {
            this->WarpFloatingImage(this->interpolation);
            GetSimilarityMeasureGradient();
        } else {
            SetGradientImageToZero();
        }
    } else GetApproximatedGradient();
    this->optimiser->IncrementCurrentIterationNumber();

    // Smooth the gradient if require
    SmoothGradient();

    if (!this->useApproxGradient) {
        // Compute the penalty term gradients if required
        GetBendingEnergyGradient();
        GetJacobianBasedGradient();
        GetLinearEnergyGradient();
        GetLandmarkDistanceGradient();
        GetInverseConsistencyGradient();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetObjectiveFunctionGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DisplayCurrentLevelParameters() {
    reg_f3d<T>::DisplayCurrentLevelParameters();
#ifdef NDEBUG
    if (this->verbose) {
#endif
        char text[255];
        reg_print_info(this->executableName, "Current backward control point image");
        sprintf(text, "\t* image dimension: %i x %i x %i",
                backwardControlPointGrid->nx, backwardControlPointGrid->ny, backwardControlPointGrid->nz);
        reg_print_info(this->executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm",
                backwardControlPointGrid->dx, backwardControlPointGrid->dy, backwardControlPointGrid->dz);
        reg_print_info(this->executableName, text);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG

    if (backwardControlPointGrid->sform_code > 0)
        reg_mat44_disp(&(backwardControlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP sform");
    else reg_mat44_disp(&(backwardControlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP qform");
#endif
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DisplayCurrentLevelParameters");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::GetInverseConsistencyErrorField(bool forceAll) {
    if (inverseConsistencyWeight <= 0) return;

    // Compute both deformation fields
    if (this->similarityWeight <= 0 || forceAll)
        GetDeformationField();
    // Compose the obtained deformation fields by the inverse transformations
    reg_spline_getDeformationField(backwardControlPointGrid,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true); // use B-Spline
    reg_spline_getDeformationField(this->controlPointGrid,
                                   backwardDeformationFieldImage,
                                   floatingMask,
                                   true, // composition
                                   true); // use B-Spline
    // Convert the deformation fields into displacement
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    reg_getDisplacementFromDeformation(backwardDeformationFieldImage);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetInverseConsistencyErrorField");
#endif
}
/* *************************************************************** */
template<class T>
double reg_f3d2<T>::GetInverseConsistencyPenaltyTerm() {
    if (inverseConsistencyWeight <= 0) return 0;

    GetInverseConsistencyErrorField(false);

    double ferror = 0;
    size_t voxelNumber = this->deformationFieldImage->nx * this->deformationFieldImage->ny * this->deformationFieldImage->nz;
    T *dispPtrX = static_cast<T*>(this->deformationFieldImage->data);
    T *dispPtrY = &dispPtrX[voxelNumber];
    if (this->deformationFieldImage->nz > 1) {
        T *dispPtrZ = &dispPtrY[voxelNumber];
        for (size_t i = 0; i < voxelNumber; ++i) {
            if (this->currentMask[i] > -1) {
                double dist = reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
                ferror += dist;
            }
        }
    } else {
        for (size_t i = 0; i < voxelNumber; ++i) {
            if (this->currentMask[i] > -1) {
                double dist = reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
                ferror += dist;
            }
        }
    }

    double berror = 0;
    voxelNumber = backwardDeformationFieldImage->nx * backwardDeformationFieldImage->ny * backwardDeformationFieldImage->nz;
    dispPtrX = static_cast<T*>(backwardDeformationFieldImage->data);
    dispPtrY = &dispPtrX[voxelNumber];
    if (backwardDeformationFieldImage->nz > 1) {
        T *dispPtrZ = &dispPtrY[voxelNumber];
        for (size_t i = 0; i < voxelNumber; ++i) {
            if (floatingMask[i] > -1) {
                double dist = reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
                berror += dist;
            }
        }
    } else {
        for (size_t i = 0; i < voxelNumber; ++i) {
            if (floatingMask[i] > -1) {
                double dist = reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
                berror += dist;
            }
        }
    }
    double error = (ferror / double(this->activeVoxelNumber[this->currentLevel]) +
                    berror / double(backwardActiveVoxelNumber[this->currentLevel]));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetInverseConsistencyPenaltyTerm");
#endif
    return double(inverseConsistencyWeight) * error;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::GetInverseConsistencyGradient() {
    if (inverseConsistencyWeight <= 0) return;

    // Note: I simplified the gradient computation in order to include
    // only d(B(F(x)))/d(forwardNode) and d(F(B(x)))/d(backwardNode)
    // I ignored d(F(B(x)))/d(forwardNode) and d(B(F(x)))/d(backwardNode)
    // cause it would only be an approximation since I don't have the
    // real inverses
    GetInverseConsistencyErrorField(true);

    // The forward inverse consistency field is masked
    size_t forwardVoxelNumber = this->deformationFieldImage->nx * this->deformationFieldImage->ny * this->deformationFieldImage->nz;
    T *defPtrX = static_cast<T*>(this->deformationFieldImage->data);
    T *defPtrY = &defPtrX[forwardVoxelNumber];
    T *defPtrZ = &defPtrY[forwardVoxelNumber];
    for (size_t i = 0; i < forwardVoxelNumber; ++i) {
        if (this->currentMask[i] < 0) {
            defPtrX[i] = 0;
            defPtrY[i] = 0;
            if (this->deformationFieldImage->nz > 1)
                defPtrZ[i] = 0;
        }
    }
    // The backward inverse consistency field is masked
    size_t backwardVoxelNumber = backwardDeformationFieldImage->nx * backwardDeformationFieldImage->ny * backwardDeformationFieldImage->nz;
    defPtrX = static_cast<T*>(backwardDeformationFieldImage->data);
    defPtrY = &defPtrX[backwardVoxelNumber];
    defPtrZ = &defPtrY[backwardVoxelNumber];
    for (size_t i = 0; i < backwardVoxelNumber; ++i) {
        if (floatingMask[i] < 0) {
            defPtrX[i] = 0;
            defPtrY[i] = 0;
            if (backwardDeformationFieldImage->nz > 1)
                defPtrZ[i] = 0;
        }
    }

    // We convolve the inverse consistency map with a cubic B-Spline kernel
    // Convolution along the x axis
    float currentNodeSpacing[3];
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dx;
    bool activeAxis[3] = {1, 0, 0};
    reg_tools_kernelConvolution(this->deformationFieldImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dy;
    activeAxis[0] = 0;
    activeAxis[1] = 1;
    reg_tools_kernelConvolution(this->deformationFieldImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the z axis if required
    if (this->voxelBasedMeasureGradient->nz > 1) {
        currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dz;
        activeAxis[1] = 0;
        activeAxis[2] = 1;
        reg_tools_kernelConvolution(this->deformationFieldImage,
                                    currentNodeSpacing,
                                    CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                    nullptr, // all volumes are active
                                    activeAxis);
    }
    // The forward inverse consistency gradient is extracted at the node position
    reg_voxelCentric2NodeCentric(this->transformationGradient,
                                 this->deformationFieldImage,
                                 2.f * inverseConsistencyWeight,
                                 true, // update the current value
                                 nullptr); // no voxel to mm conversion

    // We convolve the inverse consistency map with a cubic B-Spline kernel
    // Convolution along the x axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dx;
    activeAxis[0] = 1;
    activeAxis[1] = 0;
    activeAxis[2] = 0;
    reg_tools_kernelConvolution(backwardDeformationFieldImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dy;
    activeAxis[0] = 0;
    activeAxis[1] = 1;
    reg_tools_kernelConvolution(backwardDeformationFieldImage,
                                currentNodeSpacing,
                                CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                nullptr, // all volumes are active
                                activeAxis);
    // Convolution along the z axis if required
    if (this->voxelBasedMeasureGradient->nz > 1) {
        currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = backwardControlPointGrid->dz;
        activeAxis[1] = 0;
        activeAxis[2] = 1;
        reg_tools_kernelConvolution(backwardDeformationFieldImage,
                                    currentNodeSpacing,
                                    CUBIC_SPLINE_KERNEL, // cubic spline kernel
                                    nullptr, // all volumes are active
                                    activeAxis);
    }
    // The backward inverse consistency gradient is extracted at the node position
    reg_voxelCentric2NodeCentric(backwardTransformationGradient,
                                 backwardDeformationFieldImage,
                                 2.f * inverseConsistencyWeight,
                                 true, // update the current value
                                 nullptr); // no voxel to mm conversion

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetInverseConsistencyGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SetOptimiser() {
    if (this->useConjGradient)
        this->optimiser = new reg_conjugateGradient<T>();
    else this->optimiser = new reg_optimiser<T>();
    this->optimiser->Initialise(this->controlPointGrid->nvox,
                                this->controlPointGrid->nz > 1 ? 3 : 2,
                                this->optimiseX,
                                this->optimiseY,
                                this->optimiseZ,
                                this->maxIterationNumber,
                                0, // currentIterationNumber
                                this,
                                static_cast<T*>(this->controlPointGrid->data),
                                static_cast<T*>(this->transformationGradient->data),
                                backwardControlPointGrid->nvox,
                                static_cast<T*>(backwardControlPointGrid->data),
                                static_cast<T*>(backwardTransformationGradient->data));
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::PrintCurrentObjFunctionValue(T currentSize) {
    if (!this->verbose) return;

    char text[255];
    sprintf(text, "[%i] Current objective function: %g",
            (int)this->optimiser->GetCurrentIterationNumber(),
            this->optimiser->GetBestObjFunctionValue());
    sprintf(text + strlen(text), " = (wSIM)%g", this->bestWMeasure);
    if (this->bendingEnergyWeight > 0)
        sprintf(text + strlen(text), " - (wBE)%.2e", this->bestWBE);
    if (this->linearEnergyWeight)
        sprintf(text + strlen(text), " - (wLE)%.2e", this->bestWLE);
    if (this->jacobianLogWeight > 0)
        sprintf(text + strlen(text), " - (wJAC)%.2e", this->bestWJac);
    if (this->landmarkRegWeight > 0)
        sprintf(text + strlen(text), " - (wLAN)%.2e", this->bestWLand);
    if (inverseConsistencyWeight > 0)
        sprintf(text + strlen(text), " - (wIC)%.2e", bestIC);
    sprintf(text + strlen(text), " [+ %g mm]", currentSize);
    reg_print_info(this->executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::PrintCurrentObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::UpdateBestObjFunctionValue() {
    reg_f3d<T>::UpdateBestObjFunctionValue();
    bestIC = currentIC;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::PrintInitialObjFunctionValue() {
    if (!this->verbose) return;
    reg_f3d<T>::PrintInitialObjFunctionValue();
    //   char text[255];
    //   sprintf(text, "Initial Inverse consistency value: %g", bestIC);
    //   reg_print_info(this->executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::PrintInitialObjFunctionValue");
#endif
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::GetObjectiveFunctionValue() {
    this->currentWJac = ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

    this->currentWBE = ComputeBendingEnergyPenaltyTerm();

    this->currentWLE = ComputeLinearEnergyPenaltyTerm();

    this->currentWLand = ComputeLandmarkDistancePenaltyTerm();

    // Compute initial similarity measure
    this->currentWMeasure = 0;
    if (this->similarityWeight > 0) {
        this->WarpFloatingImage(this->interpolation);
        this->currentWMeasure = this->ComputeSimilarityMeasure();
    }

    // Compute the Inverse consistency penalty term if required
    currentIC = GetInverseConsistencyPenaltyTerm();

#ifndef NDEBUG
    char text[255];
    sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g | (wLan) %g | (wIC) %g",
            this->currentWMeasure, this->currentWBE, this->currentWLE,
            this->currentWJac, this->currentWLand, currentIC);
    reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetObjectiveFunctionValue");
#endif
    // Store the global objective function value
    return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac - currentIC;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::InitialiseSimilarity() {
    // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
    if (!this->measure_nmi && !this->measure_ssd && !this->measure_dti && !this->measure_lncc &&
        !this->measure_kld && !this->measure_mind && !this->measure_mindssc) {
        this->measure_nmi = new reg_nmi;
        for (int i = 0; i < this->inputReference->nt; ++i)
            this->measure_nmi->SetTimepointWeight(i, 1);
    }
    if (this->measure_nmi)
        this->measure_nmi->InitialiseMeasure(this->reference,
                                             this->floating,
                                             this->currentMask,
                                             this->warped,
                                             this->warpedGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent,
                                             floatingMask,
                                             backwardWarped,
                                             backwardWarpedGradientImage,
                                             backwardVoxelBasedMeasureGradientImage);

    if (this->measure_ssd)
        this->measure_ssd->InitialiseMeasure(this->reference,
                                             this->floating,
                                             this->currentMask,
                                             this->warped,
                                             this->warpedGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent,
                                             floatingMask,
                                             backwardWarped,
                                             backwardWarpedGradientImage,
                                             backwardVoxelBasedMeasureGradientImage);

    if (this->measure_kld)
        this->measure_kld->InitialiseMeasure(this->reference,
                                             this->floating,
                                             this->currentMask,
                                             this->warped,
                                             this->warpedGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent,
                                             floatingMask,
                                             backwardWarped,
                                             backwardWarpedGradientImage,
                                             backwardVoxelBasedMeasureGradientImage);

    if (this->measure_lncc)
        this->measure_lncc->InitialiseMeasure(this->reference,
                                              this->floating,
                                              this->currentMask,
                                              this->warped,
                                              this->warpedGradient,
                                              this->voxelBasedMeasureGradient,
                                              this->localWeightSimCurrent,
                                              floatingMask,
                                              backwardWarped,
                                              backwardWarpedGradientImage,
                                              backwardVoxelBasedMeasureGradientImage);

    if (this->measure_dti)
        this->measure_dti->InitialiseMeasure(this->reference,
                                             this->floating,
                                             this->currentMask,
                                             this->warped,
                                             this->warpedGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent,
                                             floatingMask,
                                             backwardWarped,
                                             backwardWarpedGradientImage,
                                             backwardVoxelBasedMeasureGradientImage);

    if (this->measure_mind)
        this->measure_mind->InitialiseMeasure(this->reference,
                                              this->floating,
                                              this->currentMask,
                                              this->warped,
                                              this->warpedGradient,
                                              this->voxelBasedMeasureGradient,
                                              this->localWeightSimCurrent,
                                              floatingMask,
                                              backwardWarped,
                                              backwardWarpedGradientImage,
                                              backwardVoxelBasedMeasureGradientImage);

    if (this->measure_mindssc)
        this->measure_mindssc->InitialiseMeasure(this->reference,
                                                 this->floating,
                                                 this->currentMask,
                                                 this->warped,
                                                 this->warpedGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 this->localWeightSimCurrent,
                                                 floatingMask,
                                                 backwardWarped,
                                                 backwardWarpedGradientImage,
                                                 backwardVoxelBasedMeasureGradientImage);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::InitialiseSimilarity");
#endif
}
/* *************************************************************** */
template<class T>
nifti_image* reg_f3d2<T>::GetBackwardControlPointPositionImage() {
    // Create a control point grid nifti image
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(backwardControlPointGrid);
    // Allocate the new image data array
    returnedControlPointGrid->data = malloc(returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    // Copy the final backward control point grid image
    memcpy(returnedControlPointGrid->data, backwardControlPointGrid->data,
           returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    // Return the new control point grid
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetBackwardControlPointPositionImage");
#endif
    return returnedControlPointGrid;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UseBCHUpdate(int v) {
    bchUpdate = true;
    useGradientCumulativeExp = false;
    bchUpdateValue = v;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UseGradientCumulativeExp() {
    bchUpdate = false;
    useGradientCumulativeExp = true;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DoNotUseGradientCumulativeExp() {
    useGradientCumulativeExp = false;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::Initialise() {
    reg_f3d<T>::Initialise();

    if (!this->inputControlPointGrid) {
        // Define the spacing for the first level
        float gridSpacing[3] = {this->spacing[0], this->spacing[1], this->spacing[2]};
        if (this->spacing[0] < 0)
            gridSpacing[0] *= -(this->inputReference->dx + this->inputFloating->dx) / 2.f;
        if (this->spacing[1] < 0)
            gridSpacing[1] *= -(this->inputReference->dy + this->inputFloating->dy) / 2.f;
        if (this->spacing[2] < 0)
            gridSpacing[2] *= -(this->inputReference->dz + this->inputFloating->dz) / 2.f;
        gridSpacing[0] *= powf(2, this->levelNumber - 1);
        gridSpacing[1] *= powf(2, this->levelNumber - 1);
        gridSpacing[2] *= powf(2, this->levelNumber - 1);

        // Create the forward and backward control point grids
        reg_createSymmetricControlPointGrids<T>(&this->controlPointGrid,
                                                &backwardControlPointGrid,
                                                this->referencePyramid[0],
                                                this->floatingPyramid[0],
                                                this->affineTransformation,
                                                gridSpacing);
    } else {
        // The control point grid image is initialised with the provided grid
        this->controlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
        this->controlPointGrid->data = malloc(this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
        if (this->inputControlPointGrid->num_ext > 0)
            nifti_copy_extensions(this->controlPointGrid, this->inputControlPointGrid);
        memcpy(this->controlPointGrid->data, this->inputControlPointGrid->data,
               this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
        // The final grid spacing is computed
        this->spacing[0] = this->controlPointGrid->dx / powf(2, this->levelNumber - 1);
        this->spacing[1] = this->controlPointGrid->dy / powf(2, this->levelNumber - 1);
        if (this->controlPointGrid->nz > 1)
            this->spacing[2] = this->controlPointGrid->dz / powf(2, this->levelNumber - 1);
        // The backward grid is derived from the forward
        backwardControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
        backwardControlPointGrid->data = malloc(backwardControlPointGrid->nvox * backwardControlPointGrid->nbyper);
        if (this->controlPointGrid->num_ext > 0)
            nifti_copy_extensions(backwardControlPointGrid, this->controlPointGrid);
        memcpy(backwardControlPointGrid->data, this->controlPointGrid->data,
               backwardControlPointGrid->nvox * backwardControlPointGrid->nbyper);
        reg_getDisplacementFromDeformation(backwardControlPointGrid);
        reg_tools_multiplyValueToImage(backwardControlPointGrid, backwardControlPointGrid, -1);
        reg_getDeformationFromDisplacement(backwardControlPointGrid);
        for (int i = 0; i < backwardControlPointGrid->num_ext; ++i) {
            mat44 tempMatrix = nifti_mat44_inverse(*reinterpret_cast<mat44 *>(backwardControlPointGrid->ext_list[i].edata));
            memcpy(backwardControlPointGrid->ext_list[i].edata, &tempMatrix, sizeof(mat44));
        }
    }

    // Set the floating mask image pyramid
    if (this->usePyramid) {
        floatingMaskPyramid = (int**)malloc(this->levelToPerform * sizeof(int*));
        backwardActiveVoxelNumber = (int*)malloc(this->levelToPerform * sizeof(int));
    } else {
        floatingMaskPyramid = (int**)malloc(sizeof(int*));
        backwardActiveVoxelNumber = (int*)malloc(sizeof(int));
    }

    if (this->usePyramid) {
        if (floatingMaskImage)
            reg_createMaskPyramid<T>(floatingMaskImage,
                                     floatingMaskPyramid,
                                     this->levelNumber,
                                     this->levelToPerform,
                                     backwardActiveVoxelNumber);
        else {
            for (unsigned int l = 0; l < this->levelToPerform; ++l) {
                backwardActiveVoxelNumber[l] = this->floatingPyramid[l]->nx * this->floatingPyramid[l]->ny * this->floatingPyramid[l]->nz;
                floatingMaskPyramid[l] = (int*)calloc(backwardActiveVoxelNumber[l], sizeof(int));
            }
        }
    } else  // no pyramid
    {
        if (floatingMaskImage)
            reg_createMaskPyramid<T>(floatingMaskImage, floatingMaskPyramid, 1, 1, backwardActiveVoxelNumber);
        else {
            backwardActiveVoxelNumber[0] = this->floatingPyramid[0]->nx * this->floatingPyramid[0]->ny * this->floatingPyramid[0]->nz;
            floatingMaskPyramid[0] = (int*)calloc(backwardActiveVoxelNumber[0], sizeof(int));
        }
    }

#ifdef NDEBUG
    if (this->verbose) {
#endif
        if (inverseConsistencyWeight > 0) {
            char text[255];
            sprintf(text, "Inverse consistency error penalty term weight: %g",
                    inverseConsistencyWeight);
            reg_print_info(this->executableName, text);
        }
#ifdef NDEBUG
    }
#endif

    // Convert the control point grid into velocity field parametrisation
    this->controlPointGrid->intent_p1 = SPLINE_VEL_GRID;
    backwardControlPointGrid->intent_p1 = SPLINE_VEL_GRID;
    // Set the number of composition to 6 by default
    this->controlPointGrid->intent_p2 = 6;
    backwardControlPointGrid->intent_p2 = 6;

#ifndef NDEBUG
    reg_print_msg_debug("reg_f3d2::Initialise() done");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::ExponentiateGradient() {
    if (!useGradientCumulativeExp) return;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
    // Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
    reg_print_msg_debug("Update the forward measure gradient using a Dartel like approach");
#endif
    // Create all deformation field images needed for resampling
    nifti_image **tempDef = (nifti_image**)malloc(size_t(fabs(backwardControlPointGrid->intent_p2) + 1) * sizeof(nifti_image*));
    for (int i = 0; i <= (int)fabs(backwardControlPointGrid->intent_p2); ++i) {
        tempDef[i] = nifti_copy_nim_info(this->deformationFieldImage);
        tempDef[i]->data = malloc(tempDef[i]->nvox * tempDef[i]->nbyper);
    }
    // Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(backwardControlPointGrid, tempDef);

    // Remove the affine component
    nifti_image *affine_disp = nullptr;
    if (this->affineTransformation) {
        affine_disp = nifti_copy_nim_info(this->deformationFieldImage);
        affine_disp->data = malloc(affine_disp->nvox * affine_disp->nbyper);
        mat44 backwardAffineTransformation = nifti_mat44_inverse(*this->affineTransformation);
        reg_affine_getDeformationField(&backwardAffineTransformation, affine_disp);
        reg_getDisplacementFromDeformation(affine_disp);
    }

    /* Allocate a temporary gradient image to store the backward gradient */
    nifti_image *tempGrad = nifti_copy_nim_info(this->voxelBasedMeasureGradient);

    tempGrad->data = malloc(tempGrad->nvox * tempGrad->nbyper);
    for (int i = 0; i < (int)fabsf(backwardControlPointGrid->intent_p2); ++i) {
        if (affine_disp)
            reg_tools_substractImageToImage(tempDef[i], affine_disp, tempDef[i]);
        reg_resampleGradient(this->voxelBasedMeasureGradient, // floating
                             tempGrad, // warped - out
                             tempDef[i], // deformation field
                             1, // interpolation type - linear
                             0); // padding value
        reg_tools_addImageToImage(tempGrad, // in1
                                  this->voxelBasedMeasureGradient, // in2
                                  this->voxelBasedMeasureGradient); // out
    }

    // Free the temporary deformation fields
    for (int i = 0; i <= (int)fabsf(backwardControlPointGrid->intent_p2); ++i) {
        nifti_image_free(tempDef[i]);
        tempDef[i] = nullptr;
    }
    free(tempDef);
    tempDef = nullptr;
    // Free the temporary gradient image
    nifti_image_free(tempGrad);
    tempGrad = nullptr;
    // Free the temporary affine displacement field
    if (affine_disp)
        nifti_image_free(affine_disp);
    affine_disp = nullptr;
    // Normalise the forward gradient
    reg_tools_divideValueToImage(this->voxelBasedMeasureGradient, // in
                                 this->voxelBasedMeasureGradient, // out
                                 powf(2, fabsf(backwardControlPointGrid->intent_p2))); // value

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
    /* Exponentiate the backward gradient using the forward transformation */
#ifndef NDEBUG
    reg_print_msg_debug("Update the backward measure gradient using a Dartel like approach");
#endif
    // Allocate a temporary gradient image to store the backward gradient
    tempGrad = nifti_copy_nim_info(backwardVoxelBasedMeasureGradientImage);
    tempGrad->data = malloc(tempGrad->nvox * tempGrad->nbyper);
    // Create all deformation field images needed for resampling
    tempDef = (nifti_image**)malloc(size_t(fabs(this->controlPointGrid->intent_p2) + 1) * sizeof(nifti_image*));
    for (int i = 0; i <= (int)fabs(this->controlPointGrid->intent_p2); ++i) {
        tempDef[i] = nifti_copy_nim_info(backwardDeformationFieldImage);
        tempDef[i]->data = malloc(tempDef[i]->nvox * tempDef[i]->nbyper);
    }
    // Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->controlPointGrid, tempDef);

    // Remove the affine component
    if (this->affineTransformation) {
        affine_disp = nifti_copy_nim_info(backwardDeformationFieldImage);
        affine_disp->data = malloc(affine_disp->nvox * affine_disp->nbyper);
        reg_affine_getDeformationField(this->affineTransformation, affine_disp);
        reg_getDisplacementFromDeformation(affine_disp);
    }

    for (int i = 0; i < (int)fabsf(this->controlPointGrid->intent_p2); ++i) {
        if (affine_disp)
            reg_tools_substractImageToImage(tempDef[i], affine_disp, tempDef[i]);
        reg_resampleGradient(backwardVoxelBasedMeasureGradientImage, // floating
                             tempGrad, // warped - out
                             tempDef[i], // deformation field
                             1, // interpolation type - linear
                             0); // padding value
        reg_tools_addImageToImage(tempGrad, // in1
                                  backwardVoxelBasedMeasureGradientImage, // in2
                                  backwardVoxelBasedMeasureGradientImage); // out
    }

    // Free the temporary deformation field
    for (int i = 0; i <= (int)fabsf(this->controlPointGrid->intent_p2); ++i) {
        nifti_image_free(tempDef[i]);
        tempDef[i] = nullptr;
    }
    free(tempDef);
    tempDef = nullptr;
    // Free the temporary gradient image
    nifti_image_free(tempGrad);
    tempGrad = nullptr;
    // Free the temporary affine displacement field
    if (affine_disp)
        nifti_image_free(affine_disp);
    affine_disp = nullptr;
    // Normalise the backward gradient
    reg_tools_divideValueToImage(backwardVoxelBasedMeasureGradientImage, // in
                                 backwardVoxelBasedMeasureGradientImage, // out
                                 powf(2, fabsf(this->controlPointGrid->intent_p2))); // value
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UpdateParameters(float scale) {
    // Restore the last successful control point grids
    this->optimiser->RestoreBestDOF();

    /************************/
    /**** Forward update ****/
    /************************/
    // Scale the gradient image
    nifti_image *forwardScaledGradient = nifti_copy_nim_info(this->transformationGradient);
    forwardScaledGradient->data = malloc(forwardScaledGradient->nvox * forwardScaledGradient->nbyper);
    reg_tools_multiplyValueToImage(this->transformationGradient,
                                   forwardScaledGradient,
                                   scale);
    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if (bchUpdate) {
        // Compute the BCH update
        reg_print_msg_warn("USING BCH FORWARD - TESTING ONLY");
#ifndef NDEBUG
        reg_print_msg_debug("Update the forward control point grid using BCH approximation");
#endif
        compute_BCH_update(this->controlPointGrid,
                           forwardScaledGradient,
                           bchUpdateValue);
    } else {
        // Reset the gradient along the axes if appropriate
        reg_setGradientToZero(forwardScaledGradient,
                              !this->optimiser->GetOptimiseX(),
                              !this->optimiser->GetOptimiseY(),
                              !this->optimiser->GetOptimiseZ());
        // Update the velocity field
        reg_tools_addImageToImage(this->controlPointGrid, // in1
                                  forwardScaledGradient, // in2
                                  this->controlPointGrid); // out
    }
    // Clean the temporary nifti_images
    nifti_image_free(forwardScaledGradient);
    forwardScaledGradient = nullptr;

    /************************/
    /**** Backward update ***/
    /************************/
    // Scale the gradient image
    nifti_image *backwardScaledGradient = nifti_copy_nim_info(backwardTransformationGradient);
    backwardScaledGradient->data = malloc(backwardScaledGradient->nvox * backwardScaledGradient->nbyper);
    reg_tools_multiplyValueToImage(backwardTransformationGradient,
                                   backwardScaledGradient,
                                   scale);
    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if (bchUpdate) {
        // Compute the BCH update
        reg_print_msg_warn("USING BCH BACKWARD - TESTING ONLY");
#ifndef NDEBUG
        reg_print_msg_debug("Update the backward control point grid using BCH approximation");
#endif
        compute_BCH_update(backwardControlPointGrid,
                           backwardScaledGradient,
                           bchUpdateValue);
    } else {
        // Reset the gradient along the axes if appropriate
        reg_setGradientToZero(backwardScaledGradient,
                              !this->optimiser->GetOptimiseX(),
                              !this->optimiser->GetOptimiseY(),
                              !this->optimiser->GetOptimiseZ());
        // Update the velocity field
        reg_tools_addImageToImage(backwardControlPointGrid, // in1
                                  backwardScaledGradient, // in2
                                  backwardControlPointGrid); // out
    }
    // Clean the temporary nifti_images
    nifti_image_free(backwardScaledGradient);
    backwardScaledGradient = nullptr;

    /****************************/
    /******** Symmetrise ********/
    /****************************/

    // In order to ensure symmetry the forward and backward velocity fields
    // are averaged in both image spaces: reference and floating
    /****************************/
    nifti_image *warpedForwardTrans = nifti_copy_nim_info(backwardControlPointGrid);
    warpedForwardTrans->data = malloc(warpedForwardTrans->nvox * warpedForwardTrans->nbyper);
    nifti_image *warpedBackwardTrans = nifti_copy_nim_info(this->controlPointGrid);
    warpedBackwardTrans->data = malloc(warpedBackwardTrans->nvox * warpedBackwardTrans->nbyper);

    // Both parametrisations are converted into displacement
    reg_getDisplacementFromDeformation(this->controlPointGrid);
    reg_getDisplacementFromDeformation(backwardControlPointGrid);

    // Both parametrisations are copied over
    memcpy(warpedBackwardTrans->data, backwardControlPointGrid->data, warpedBackwardTrans->nvox * warpedBackwardTrans->nbyper);
    memcpy(warpedForwardTrans->data, this->controlPointGrid->data, warpedForwardTrans->nvox * warpedForwardTrans->nbyper);

    // and subtracted (sum and negation)
    reg_tools_substractImageToImage(backwardControlPointGrid, // displacement
                                    warpedForwardTrans, // displacement
                                    backwardControlPointGrid); // displacement output
    reg_tools_substractImageToImage(this->controlPointGrid, // displacement
                                    warpedBackwardTrans, // displacement
                                    this->controlPointGrid); // displacement output
    // Division by 2
    reg_tools_multiplyValueToImage(backwardControlPointGrid, // displacement
                                   backwardControlPointGrid, // displacement
                                   0.5f);
    reg_tools_multiplyValueToImage(this->controlPointGrid, // displacement
                                   this->controlPointGrid, // displacement
                                   0.5f);
    // Clean the temporary allocated velocity fields
    nifti_image_free(warpedForwardTrans);
    warpedForwardTrans = nullptr;
    nifti_image_free(warpedBackwardTrans);
    warpedBackwardTrans = nullptr;

    // Convert the velocity field from displacement to deformation
    reg_getDeformationFromDisplacement(this->controlPointGrid);
    reg_getDeformationFromDisplacement(backwardControlPointGrid);
}
/* *************************************************************** */
template<class T>
nifti_image** reg_f3d2<T>::GetWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !this->controlPointGrid || !backwardControlPointGrid) {
        reg_print_fct_error("reg_f3d2<T>::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    // Set the input images
    reg_f3d2<T>::reference = this->inputReference;
    reg_f3d2<T>::floating = this->inputFloating;
    // No mask is used to perform the final resampling
    reg_f3d2<T>::currentMask = nullptr;
    reg_f3d2<T>::floatingMask = nullptr;

    // Allocate the forward and backward warped images
    AllocateWarped();
    // Allocate the forward and backward dense deformation field
    AllocateDeformationField();

    // Warp the floating images into the reference spaces using a cubic spline interpolation
    reg_f3d2<T>::WarpFloatingImage(3); // cubic spline interpolation

    // Deallocate the deformation field
    DeallocateDeformationField();

    // Allocate and save the forward transformation warped image
    nifti_image **warpedImage = (nifti_image**)malloc(2 * sizeof(nifti_image*));
    warpedImage[0] = nifti_copy_nim_info(this->warped);
    warpedImage[0]->cal_min = this->inputFloating->cal_min;
    warpedImage[0]->cal_max = this->inputFloating->cal_max;
    warpedImage[0]->scl_slope = this->inputFloating->scl_slope;
    warpedImage[0]->scl_inter = this->inputFloating->scl_inter;
    warpedImage[0]->data = malloc(warpedImage[0]->nvox * warpedImage[0]->nbyper);
    memcpy(warpedImage[0]->data, this->warped->data, warpedImage[0]->nvox * warpedImage[0]->nbyper);

    // Allocate and save the backward transformation warped image
    warpedImage[1] = nifti_copy_nim_info(backwardWarped);
    warpedImage[1]->cal_min = this->inputReference->cal_min;
    warpedImage[1]->cal_max = this->inputReference->cal_max;
    warpedImage[1]->scl_slope = this->inputReference->scl_slope;
    warpedImage[1]->scl_inter = this->inputReference->scl_inter;
    warpedImage[1]->data = malloc(warpedImage[1]->nvox * warpedImage[1]->nbyper);
    memcpy(warpedImage[1]->data, backwardWarped->data, warpedImage[1]->nvox * warpedImage[1]->nbyper);

    // Deallocate the warped images
    DeallocateWarped();

    // Return the two final warped images
    return warpedImage;
}
/* *************************************************************** */
template class reg_f3d2<float>;
