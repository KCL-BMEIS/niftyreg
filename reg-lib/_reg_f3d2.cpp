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
#include "F3dContent.h"

#ifdef _USE_CUDA
#include "CudaF3dContent.h"
#endif

/* *************************************************************** */
template <class T>
reg_f3d2<T>::reg_f3d2(int refTimePoint, int floTimePoint):
    reg_f3d<T>::reg_f3d(refTimePoint, floTimePoint) {
    this->executableName = (char*)"NiftyReg F3D2";
    controlPointGridBw = nullptr;
    floatingMaskImage = nullptr;
    floatingMaskPyramid = nullptr;
    affineTransformationBw = nullptr;
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
    if (controlPointGridBw) {
        nifti_image_free(controlPointGridBw);
        controlPointGridBw = nullptr;
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

    if (affineTransformationBw) {
        delete affineTransformationBw;
        affineTransformationBw = nullptr;
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
template<class T>
void reg_f3d2<T>::InitContent(nifti_image *reference, nifti_image *floating, int *mask) {
    if (this->platformType == PlatformType::Cpu)
        conBw = new F3dContent(floating, reference, controlPointGridBw, nullptr, mask, affineTransformationBw, sizeof(T));
#ifdef _USE_CUDA
    else if (this->platformType == PlatformType::Cuda)
        conBw = new CudaF3dContent(floating, reference, controlPointGridBw, nullptr, mask, affineTransformationBw, sizeof(T));
#endif
    computeBw = this->platform->CreateCompute(*conBw);
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::InitCurrentLevel(int currentLevel) {
    // Set the current input images
    nifti_image *reference, *floating;
    int *referenceMask, *floatingMask;
    if (currentLevel < 0) {
        reference = this->inputReference;
        floating = this->inputFloating;
        referenceMask = nullptr;
        floatingMask = nullptr;
    } else {
        const int index = this->usePyramid ? currentLevel : 0;
        reference = this->referencePyramid[index];
        floating = this->floatingPyramid[index];
        referenceMask = this->maskPyramid[index];
        floatingMask = floatingMaskPyramid[index];
    }

    // Define the initial step size for the gradient ascent optimisation
    T maxStepSize = reference->dx;
    maxStepSize = reference->dy > maxStepSize ? reference->dy : maxStepSize;
    maxStepSize = floating->dx > maxStepSize ? floating->dx : maxStepSize;
    maxStepSize = floating->dy > maxStepSize ? floating->dy : maxStepSize;
    if (reference->ndim > 2) {
        maxStepSize = (reference->dz > maxStepSize) ? reference->dz : maxStepSize;
        maxStepSize = (floating->dz > maxStepSize) ? floating->dz : maxStepSize;
    }

    // Refine the control point grids if required
    // Don't if currentLevel < 0, since it's not required for GetWarpedImage()
    if (this->gridRefinement && currentLevel >= 0) {
        if (currentLevel == 0) {
            this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16, this->levelNumber - 1));
            this->linearEnergyWeight = this->linearEnergyWeight / static_cast<T>(powf(3, this->levelNumber - 1));
        } else {
            this->bendingEnergyWeight = this->bendingEnergyWeight * 16;
            this->linearEnergyWeight = this->linearEnergyWeight * 3;
            reg_spline_refineControlPointGrid(this->controlPointGrid);
            reg_spline_refineControlPointGrid(controlPointGridBw);
        }
    }

    reg_f3d<T>::InitContent(reference, floating, referenceMask);
    InitContent(reference, floating, floatingMask);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::InitCurrentLevel");
#endif
    return maxStepSize;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::DeinitCurrentLevel(int currentLevel) {
    reg_f3d<T>::DeinitCurrentLevel(currentLevel);
    delete computeBw;
    computeBw = nullptr;
    delete conBw;
    conBw = nullptr;
    if (currentLevel >= 0) {
        if (this->usePyramid) {
            free(floatingMaskPyramid[currentLevel]);
            floatingMaskPyramid[currentLevel] = nullptr;
        } else if (currentLevel == this->levelToPerform - 1) {
            free(floatingMaskPyramid[0]);
            floatingMaskPyramid[0] = nullptr;
        }
    }
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
    // By default the number of steps is automatically updated
    bool updateStepNumber = true;
    if (!this->optimiser)
        updateStepNumber = false;

#ifndef NDEBUG
    char text[255];
    sprintf(text, "Velocity integration forward. Step number update=%i", updateStepNumber);
    reg_print_msg_debug(text);
#endif
    // The forward transformation is computed using the scaling-and-squaring approach
    this->compute->GetDefFieldFromVelocityGrid(updateStepNumber);

#ifndef NDEBUG
    sprintf(text, "Velocity integration backward. Step number update=%i", updateStepNumber);
    reg_print_msg_debug(text);
#endif
    // The number of step number is copied over from the forward transformation
    controlPointGridBw->intent_p2 = this->controlPointGrid->intent_p2;
    // The backward transformation is computed using the scaling-and-squaring approach
    computeBw->GetDefFieldFromVelocityGrid(false);
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::WarpFloatingImage(int inter) {
    reg_f3d<T>::WarpFloatingImage(inter);

    // Resample the reference image
    if (!this->measure_dti) {
        computeBw->ResampleImage(inter, this->warpedPaddingValue);
    } else {
        // reg_defField_getJacobianMatrix(backwardDeformationFieldImage, backwardJacobianMatrix);
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

    double backwardPenaltyTerm = computeBw->GetJacobianPenaltyTerm(approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (backwardPenaltyTerm != backwardPenaltyTerm && it < maxit) {
        backwardPenaltyTerm = computeBw->CorrectFolding(approx);
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
    double backwardPenaltyTerm = this->bendingEnergyWeight * computeBw->ApproxBendingEnergy();

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ComputeBendingEnergyPenaltyTerm");
#endif
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeLinearEnergyPenaltyTerm() {
    if (this->linearEnergyWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();
    double backwardPenaltyTerm = this->linearEnergyWeight * computeBw->ApproxLinearEnergy();

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
    double backwardPenaltyTerm = this->landmarkRegWeight * computeBw->GetLandmarkDistance(this->landmarkRegNumber,
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
    dynamic_cast<F3dContent*>(this->con)->ZeroVoxelBasedMeasureGradient();
    conBw->ZeroVoxelBasedMeasureGradient();

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

    for (int t = 0; t < this->con->Content::GetReference()->nt; ++t) {
        this->compute->GetImageGradient(this->interpolation, this->warpedPaddingValue, t);
        computeBw->GetImageGradient(this->interpolation, this->warpedPaddingValue, t);

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
    }

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

    // The voxel-based sim-measure-gradient is convolved with a spline kernel
    // And the backward-node-based NMI gradient is extracted
    computeBw->ConvolveVoxelBasedMeasureGradient(this->similarityWeight);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetSimilarityMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetJacobianBasedGradient() {
    if (this->jacobianLogWeight <= 0) return;

    reg_f3d<T>::GetJacobianBasedGradient();
    computeBw->JacobianPenaltyTermGradient(this->jacobianLogWeight, this->jacobianLogApproximation);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetBendingEnergyGradient() {
    if (this->bendingEnergyWeight <= 0) return;

    reg_f3d<T>::GetBendingEnergyGradient();
    computeBw->ApproxBendingEnergyGradient(this->bendingEnergyWeight);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetBendingEnergyGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetLinearEnergyGradient() {
    if (this->linearEnergyWeight <= 0) return;

    reg_f3d<T>::GetLinearEnergyGradient();
    computeBw->ApproxLinearEnergyGradient(this->linearEnergyWeight);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetLinearEnergyGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetLandmarkDistanceGradient() {
    if (this->landmarkRegWeight <= 0) return;

    reg_f3d<T>::GetLandmarkDistanceGradient();
    computeBw->LandmarkDistanceGradient(this->landmarkRegNumber,
                                        this->landmarkFloating,
                                        this->landmarkReference,
                                        this->landmarkRegWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetLandmarkDistanceGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SmoothGradient() {
    reg_f3d<T>::SmoothGradient();

    // The gradient is smoothed using a Gaussian kernel if it is required
    computeBw->SmoothGradient(this->gradientSmoothingSigma);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetApproximatedGradient() {
    reg_f3d<T>::GetApproximatedGradient();

    computeBw->GetApproximatedGradient(*this);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::NormaliseGradient() {
    // The forward gradient max length is computed
    const T forwardMaxGradLength = reg_f3d<T>::NormaliseGradient();

    // The backward gradient max length is computed
    const T backwardMaxGradLength = (T)computeBw->GetMaximalLength(this->optimiser->GetVoxNumber_b(),
                                                                   this->optimiseX,
                                                                   this->optimiseY,
                                                                   this->optimiseZ);

    // The largest value between the forward and backward gradient is kept
    const T maxGradLength = std::max(backwardMaxGradLength, forwardMaxGradLength);

#ifndef NDEBUG
    char text[255];
    sprintf(text, "Objective function gradient maximal length: %g", maxGradLength);
    reg_print_msg_debug(text);
#endif

    // The forward gradient is normalised
    this->compute->NormaliseGradient(this->optimiser->GetVoxNumber(), maxGradLength);
    // The backward gradient is normalised
    computeBw->NormaliseGradient(this->optimiser->GetVoxNumber_b(), maxGradLength);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::NormaliseGradient");
#endif
    // Returns the largest gradient distance
    return maxGradLength;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::GetObjectiveFunctionGradient() {
    if (!this->useApproxGradient) {
        // Compute the gradient of the similarity measure
        if (this->similarityWeight > 0) {
            WarpFloatingImage(this->interpolation);
            GetSimilarityMeasureGradient();
        } else {
            dynamic_cast<F3dContent*>(this->con)->ZeroTransformationGradient();
            conBw->ZeroTransformationGradient();
        }
    } else GetApproximatedGradient();
    this->optimiser->IncrementCurrentIterationNumber();

    // Smooth the gradient if require
    SmoothGradient();

    // Compute the penalty term gradients if required
    if (!this->useApproxGradient) {
        GetBendingEnergyGradient();
        GetJacobianBasedGradient();
        GetLinearEnergyGradient();
        GetLandmarkDistanceGradient();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetObjectiveFunctionGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DisplayCurrentLevelParameters(int currentLevel) {
    reg_f3d<T>::DisplayCurrentLevelParameters(currentLevel);
#ifdef NDEBUG
    if (this->verbose) {
#endif
        char text[255];
        reg_print_info(this->executableName, "Current backward control point image");
        sprintf(text, "\t* image dimension: %i x %i x %i",
                controlPointGridBw->nx, controlPointGridBw->ny, controlPointGridBw->nz);
        reg_print_info(this->executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm",
                controlPointGridBw->dx, controlPointGridBw->dy, controlPointGridBw->dz);
        reg_print_info(this->executableName, text);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    if (controlPointGridBw->sform_code > 0)
        reg_mat44_disp(&controlPointGridBw->sto_xyz, (char*)"[NiftyReg DEBUG] Backward CPP sform");
    else reg_mat44_disp(&controlPointGridBw->qto_xyz, (char*)"[NiftyReg DEBUG] Backward CPP qform");
#endif
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::DisplayCurrentLevelParameters");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SetOptimiser() {
    this->optimiser = this->platform->template CreateOptimiser<T>(*dynamic_cast<F3dContent*>(this->con),
                                                                  *this,
                                                                  this->maxIterationNumber,
                                                                  this->useConjGradient,
                                                                  this->optimiseX,
                                                                  this->optimiseY,
                                                                  this->optimiseZ,
                                                                  conBw);
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
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::PrintInitialObjFunctionValue() {
    if (!this->verbose) return;
    reg_f3d<T>::PrintInitialObjFunctionValue();
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
        WarpFloatingImage(this->interpolation);
        this->currentWMeasure = this->ComputeSimilarityMeasure();
    }

#ifndef NDEBUG
    char text[255];
    sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g | (wLan) %g",
            this->currentWMeasure, this->currentWBE, this->currentWLE,
            this->currentWJac, this->currentWLand);
    reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetObjectiveFunctionValue");
#endif
    // Store the global objective function value
    return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::InitialiseSimilarity() {
    F3dContent& con = *dynamic_cast<F3dContent*>(this->con);

    if (this->measure_nmi)
        this->measure->Initialise(*this->measure_nmi, con, conBw);

    if (this->measure_ssd)
        this->measure->Initialise(*this->measure_ssd, con, conBw);

    if (this->measure_kld)
        this->measure->Initialise(*this->measure_kld, con, conBw);

    if (this->measure_lncc)
        this->measure->Initialise(*this->measure_lncc, con, conBw);

    if (this->measure_dti)
        this->measure->Initialise(*this->measure_dti, con, conBw);

    if (this->measure_mind)
        this->measure->Initialise(*this->measure_mind, con, conBw);

    if (this->measure_mindssc)
        this->measure->Initialise(*this->measure_mindssc, con, conBw);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::InitialiseSimilarity");
#endif
}
/* *************************************************************** */
template<class T>
nifti_image* reg_f3d2<T>::GetBackwardControlPointPositionImage() {
    // Create a control point grid nifti image
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(controlPointGridBw);
    // Allocate the new image data array
    returnedControlPointGrid->data = malloc(returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    // Copy the final backward control point grid image
    memcpy(returnedControlPointGrid->data, controlPointGridBw->data,
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
                                                &controlPointGridBw,
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
        controlPointGridBw = nifti_copy_nim_info(this->controlPointGrid);
        controlPointGridBw->data = malloc(controlPointGridBw->nvox * controlPointGridBw->nbyper);
        if (this->controlPointGrid->num_ext > 0)
            nifti_copy_extensions(controlPointGridBw, this->controlPointGrid);
        memcpy(controlPointGridBw->data, this->controlPointGrid->data,
               controlPointGridBw->nvox * controlPointGridBw->nbyper);
        reg_getDisplacementFromDeformation(controlPointGridBw);
        reg_tools_multiplyValueToImage(controlPointGridBw, controlPointGridBw, -1);
        reg_getDeformationFromDisplacement(controlPointGridBw);
        for (int i = 0; i < controlPointGridBw->num_ext; ++i) {
            mat44 tempMatrix = nifti_mat44_inverse(*reinterpret_cast<mat44 *>(controlPointGridBw->ext_list[i].edata));
            memcpy(controlPointGridBw->ext_list[i].edata, &tempMatrix, sizeof(mat44));
        }
    }

    // Set the floating mask image pyramid
    if (this->usePyramid) {
        floatingMaskPyramid = (int**)malloc(this->levelToPerform * sizeof(int*));
    } else {
        floatingMaskPyramid = (int**)malloc(sizeof(int*));
    }

    if (this->usePyramid) {
        if (floatingMaskImage) {
            reg_createMaskPyramid<T>(floatingMaskImage, floatingMaskPyramid, this->levelNumber, this->levelToPerform);
        } else {
            for (unsigned int l = 0; l < this->levelToPerform; ++l) {
                const size_t voxelNumberBw = this->floatingPyramid[l]->nx * this->floatingPyramid[l]->ny * this->floatingPyramid[l]->nz;
                floatingMaskPyramid[l] = (int*)calloc(voxelNumberBw, sizeof(int));
            }
        }
    } else {  // no pyramid
        if (floatingMaskImage)
            reg_createMaskPyramid<T>(floatingMaskImage, floatingMaskPyramid, 1, 1);
        else {
            const size_t voxelNumberBw = this->floatingPyramid[0]->nx * this->floatingPyramid[0]->ny * this->floatingPyramid[0]->nz;
            floatingMaskPyramid[0] = (int*)calloc(voxelNumberBw, sizeof(int));
        }
    }

#ifdef NDEBUG
    if (this->verbose) {
#endif
        if (inverseConsistencyWeight > 0) {
            char text[255];
            sprintf(text, "Inverse consistency error penalty term weight: %g", inverseConsistencyWeight);
            reg_print_info(this->executableName, text);
        }
#ifdef NDEBUG
    }
#endif

    // Convert the control point grid into velocity field parametrisation
    this->controlPointGrid->intent_p1 = SPLINE_VEL_GRID;
    controlPointGridBw->intent_p1 = SPLINE_VEL_GRID;
    // Set the number of composition to 6 by default
    this->controlPointGrid->intent_p2 = controlPointGridBw->intent_p2 = 6;

    if (this->affineTransformation)
        affineTransformationBw = new mat44(nifti_mat44_inverse(*this->affineTransformation));

#ifndef NDEBUG
    reg_print_msg_debug("reg_f3d2::Initialise() done");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::ExponentiateGradient() {
    if (!useGradientCumulativeExp) return;

    // Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
    reg_print_msg_debug("Update the forward measure gradient using a Dartel like approach");
#endif
    this->compute->ExponentiateGradient(*conBw);

    /* Exponentiate the backward gradient using the forward transformation */
#ifndef NDEBUG
    reg_print_msg_debug("Update the backward measure gradient using a Dartel like approach");
#endif
    computeBw->ExponentiateGradient(*this->con);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::ExponentiateGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UpdateParameters(float scale) {
    // Restore the last successful control point grids
    this->optimiser->RestoreBestDOF();

    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if (bchUpdate) {
        // Forward update
        reg_print_msg_warn("USING BCH FORWARD - TESTING ONLY");
#ifndef NDEBUG
        reg_print_msg_debug("Update the forward control point grid using BCH approximation");
#endif
        this->compute->BchUpdate(scale, bchUpdateValue);

        // Backward update
        reg_print_msg_warn("USING BCH BACKWARD - TESTING ONLY");
#ifndef NDEBUG
        reg_print_msg_debug("Update the backward control point grid using BCH approximation");
#endif
        computeBw->BchUpdate(scale, bchUpdateValue);
    } else {
        // Forward update
        this->compute->UpdateVelocityField(scale,
                                           this->optimiser->GetOptimiseX(),
                                           this->optimiser->GetOptimiseY(),
                                           this->optimiser->GetOptimiseZ());
        // Backward update
        computeBw->UpdateVelocityField(scale,
                                       this->optimiser->GetOptimiseX(),
                                       this->optimiser->GetOptimiseY(),
                                       this->optimiser->GetOptimiseZ());
    }

    // Symmetrise
    this->compute->SymmetriseVelocityFields(*conBw);
}
/* *************************************************************** */
template<class T>
nifti_image** reg_f3d2<T>::GetWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !this->controlPointGrid || !controlPointGridBw) {
        reg_print_fct_error("reg_f3d2<T>::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    InitCurrentLevel(-1);

    WarpFloatingImage(3); // cubic spline interpolation

    F3dContent *con = dynamic_cast<F3dContent*>(this->con);
    nifti_image **warpedImage = (nifti_image**)calloc(2, sizeof(nifti_image*));
    warpedImage[0] = con->GetWarped();
    warpedImage[1] = conBw->GetWarped();

    con->SetWarped(nullptr); // Prevent deallocating of warpedImage
    conBw->SetWarped(nullptr);
    DeinitCurrentLevel(-1);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2<T>::GetWarpedImage");
#endif
    return warpedImage;
}
/* *************************************************************** */
template class reg_f3d2<float>;
