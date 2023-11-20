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

/* *************************************************************** */
template <class T>
reg_f3d2<T>::reg_f3d2(int refTimePoints, int floTimePoints):
    reg_f3d<T>::reg_f3d(refTimePoints, floTimePoints) {
    this->executableName = (char*)"NiftyReg F3D2";
    inverseConsistencyWeight = 0;
    bchUpdate = false;
    useGradientCumulativeExp = true;
    bchUpdateValue = 0;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::SetFloatingMask(NiftiImage floatingMaskImageIn) {
    floatingMaskImage = floatingMaskImageIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::SetInverseConsistencyWeight(T w) {
    inverseConsistencyWeight = w;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::InitContent(nifti_image *reference, nifti_image *floating, int *referenceMask, int *floatingMask) {
    unique_ptr<F3d2ContentCreator> contentCreator{ dynamic_cast<F3d2ContentCreator*>(this->platform->CreateContentCreator(ContentType::F3d2)) };
    auto&& [con, conBw] = contentCreator->Create(reference, floating, this->controlPointGrid, controlPointGridBw,
                                                 this->localWeightSimInput, referenceMask, floatingMask,
                                                 this->affineTransformation.get(), affineTransformationBw.get(), sizeof(T));
    this->con.reset(con);
    this->conBw.reset(conBw);
    this->compute.reset(this->platform->CreateCompute(*con));
    this->computeBw.reset(this->platform->CreateCompute(*conBw));
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::InitCurrentLevel(int currentLevel) {
    // Set the current input images
    nifti_image *reference, *floating;
    int *referenceMask, *floatingMask;
    if (currentLevel < 0) {
        // Settings for GetWarpedImage()
        // Use CPU for warping since CUDA isn't supporting Cubic interpolation
        // TODO Remove this when CUDA supports Cubic interpolation
        this->SetPlatformType(PlatformType::Cpu);
        reference = this->inputReference;
        floating = this->inputFloating;
        referenceMask = nullptr;
        floatingMask = nullptr;
    } else {
        const int index = this->usePyramid ? currentLevel : 0;
        reference = this->referencePyramid[index];
        floating = this->floatingPyramid[index];
        referenceMask = this->maskPyramid[index].get();
        floatingMask = floatingMaskPyramid[index].get();
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

    InitContent(reference, floating, referenceMask, floatingMask);

    NR_FUNC_CALLED();
    return maxStepSize;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::DeinitCurrentLevel(int currentLevel) {
    reg_f3d<T>::DeinitCurrentLevel(currentLevel);
    computeBw = nullptr;
    conBw = nullptr;
    if (currentLevel >= 0) {
        if (this->usePyramid) {
            floatingMaskPyramid[currentLevel] = nullptr;
        } else if (currentLevel == this->levelToPerform - 1) {
            floatingMaskPyramid[0] = nullptr;
        }
    }
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::CheckParameters() {
    reg_f3d<T>::CheckParameters();

    // CHECK THE FLOATING MASK DIMENSION IF IT IS DEFINED
    if (floatingMaskImage && (this->inputFloating->nx != floatingMaskImage->nx ||
                              this->inputFloating->ny != floatingMaskImage->ny ||
                              this->inputFloating->nz != floatingMaskImage->nz))
        NR_FATAL_ERROR("The floating image and its mask have different dimension");

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

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetDeformationField() {
    // By default the number of steps is automatically updated
    bool updateStepNumber = true;
    if (!this->optimiser)
        updateStepNumber = false;

    NR_DEBUG("Velocity integration forward. Step number update=" << updateStepNumber);
    // The forward transformation is computed using the scaling-and-squaring approach
    this->compute->GetDefFieldFromVelocityGrid(updateStepNumber);

    NR_DEBUG("Velocity integration backward. Step number update=" << updateStepNumber);
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
                          this->measure_dti->GetActiveTimePoints(),
                          backwardJacobianMatrix);*/
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeJacobianBasedPenaltyTerm(int type) {
    if (this->jacobianLogWeight <= 0) return 0;

    double forwardPenaltyTerm = reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(type);

    bool approx = type == 2 ? false : this->jacobianLogApproximation;

    double backwardPenaltyTerm = computeBw->GetJacobianPenaltyTerm(approx);

    unsigned maxit = 5;
    if (type > 0) maxit = 20;
    unsigned it = 0;
    while (backwardPenaltyTerm != backwardPenaltyTerm && it < maxit) {
        backwardPenaltyTerm = computeBw->CorrectFolding(approx);
        NR_DEBUG("Folding correction - Backward transformation");
        it++;
    }
    if (type > 0 && it > 0) {
        if (backwardPenaltyTerm != backwardPenaltyTerm) {
            this->optimiser->RestoreBestDof();
            NR_DEBUG("The backward transformation folding correction scheme failed");
        } else {
            NR_VERBOSE("Backward transformation folding correction, " << it << " step(s)");
        }
    }
    backwardPenaltyTerm *= this->jacobianLogWeight;

    NR_FUNC_CALLED();
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeBendingEnergyPenaltyTerm() {
    if (this->bendingEnergyWeight <= 0) return 0;
    const double forwardPenaltyTerm = reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();
    const double backwardPenaltyTerm = this->bendingEnergyWeight * computeBw->ApproxBendingEnergy();
    NR_FUNC_CALLED();
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeLinearEnergyPenaltyTerm() {
    if (this->linearEnergyWeight <= 0) return 0;
    const double forwardPenaltyTerm = reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();
    const double backwardPenaltyTerm = this->linearEnergyWeight * computeBw->ApproxLinearEnergy();
    NR_FUNC_CALLED();
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d2<T>::ComputeLandmarkDistancePenaltyTerm() {
    if (this->landmarkRegWeight <= 0) return 0;
    const double forwardPenaltyTerm = reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm();
    const double backwardPenaltyTerm = this->landmarkRegWeight * computeBw->GetLandmarkDistance(this->landmarkRegNumber,
                                                                                                this->landmarkFloating,
                                                                                                this->landmarkReference);
    NR_FUNC_CALLED();
    return forwardPenaltyTerm + backwardPenaltyTerm;
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient() {
    // The voxel based gradient image is initialised with zeros
    dynamic_cast<F3dContent&>(*this->con).ZeroVoxelBasedMeasureGradient();
    conBw->ZeroVoxelBasedMeasureGradient();

    // The intensity gradient is first computed
    //    if(this->measure_dti){
    //        reg_getImageGradient(this->floating,
    //                             this->warpedGradient,
    //                             this->deformationFieldImage,
    //                             this->currentMask,
    //                             this->interpolation,
    //                             this->warpedPaddingValue,
    //                             this->measure_dti->GetActiveTimePoints(),
    //                             this->forwardJacobianMatrix,
    //                             this->warped);

    //        reg_getImageGradient(this->reference,
    //                             backwardWarpedGradientImage,
    //                             backwardDeformationFieldImage,
    //                             floatingMask,
    //                             this->interpolation,
    //                             this->warpedPaddingValue,
    //                             this->measure_dti->GetActiveTimePoints(),
    //                             backwardJacobianMatrix,
    //                             backwardWarped);
    //   if(this->measure_dti)
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

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetSimilarityMeasureGradient() {
    reg_f3d<T>::GetSimilarityMeasureGradient();

    // The voxel-based sim-measure-gradient is convolved with a spline kernel
    // And the backward-node-based NMI gradient is extracted
    computeBw->ConvolveVoxelBasedMeasureGradient(this->similarityWeight);

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetJacobianBasedGradient() {
    if (this->jacobianLogWeight <= 0) return;
    reg_f3d<T>::GetJacobianBasedGradient();
    computeBw->JacobianPenaltyTermGradient(this->jacobianLogWeight, this->jacobianLogApproximation);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetBendingEnergyGradient() {
    if (this->bendingEnergyWeight <= 0) return;
    reg_f3d<T>::GetBendingEnergyGradient();
    computeBw->ApproxBendingEnergyGradient(this->bendingEnergyWeight);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetLinearEnergyGradient() {
    if (this->linearEnergyWeight <= 0) return;
    reg_f3d<T>::GetLinearEnergyGradient();
    computeBw->ApproxLinearEnergyGradient(this->linearEnergyWeight);
    NR_FUNC_CALLED();
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
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SmoothGradient() {
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (this->gradientSmoothingSigma == 0) return;
    reg_f3d<T>::SmoothGradient();
    computeBw->SmoothGradient(this->gradientSmoothingSigma);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetApproximatedGradient() {
    reg_f3d<T>::GetApproximatedGradient();
    computeBw->GetApproximatedGradient(*this);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
T reg_f3d2<T>::NormaliseGradient() {
    // The forward gradient max length is computed
    const T forwardMaxGradLength = reg_f3d<T>::NormaliseGradient();

    // The backward gradient max length is computed
    const T backwardMaxGradLength = (T)computeBw->GetMaximalLength(this->optimiseX,
                                                                   this->optimiseY,
                                                                   this->optimiseZ);

    // The largest value between the forward and backward gradient is kept
    const T maxGradLength = std::max(backwardMaxGradLength, forwardMaxGradLength);
    NR_DEBUG("Objective function gradient maximal length: " << maxGradLength);

    // The forward gradient is normalised
    this->compute->NormaliseGradient(maxGradLength, this->optimiseX, this->optimiseY, this->optimiseZ);
    // The backward gradient is normalised
    computeBw->NormaliseGradient(maxGradLength, this->optimiseX, this->optimiseY, this->optimiseZ);

    NR_FUNC_CALLED();
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
            dynamic_cast<F3dContent&>(*this->con).ZeroTransformationGradient();
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
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DisplayCurrentLevelParameters(int currentLevel) {
    reg_f3d<T>::DisplayCurrentLevelParameters(currentLevel);
    NR_VERBOSE("Current backward control point image");
    NR_VERBOSE("\t* image dimension: " << controlPointGridBw->nx << " x " << controlPointGridBw->ny << " x " << controlPointGridBw->nz);
    NR_VERBOSE("\t* image spacing: " << controlPointGridBw->dx << " x " << controlPointGridBw->dy << " x " << controlPointGridBw->dz << " mm");

    if (controlPointGridBw->sform_code > 0)
        NR_MAT44_DEBUG(controlPointGridBw->sto_xyz, "Backward CPP sform");
    else NR_MAT44_DEBUG(controlPointGridBw->qto_xyz, "Backward CPP qform");

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::SetOptimiser() {
    this->optimiser.reset(this->platform->template CreateOptimiser<T>(dynamic_cast<F3dContent&>(*this->con),
                                                                      *this,
                                                                      this->maxIterationNumber,
                                                                      this->useConjGradient,
                                                                      this->optimiseX,
                                                                      this->optimiseY,
                                                                      this->optimiseZ,
                                                                      conBw.get()));
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::PrintCurrentObjFunctionValue(T currentSize) {
    reg_f3d<T>::PrintCurrentObjFunctionValue(currentSize);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::UpdateBestObjFunctionValue() {
    reg_f3d<T>::UpdateBestObjFunctionValue();
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::PrintInitialObjFunctionValue() {
    reg_f3d<T>::PrintInitialObjFunctionValue();
    NR_FUNC_CALLED();
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

    NR_DEBUG("(wMeasure) " << this->currentWMeasure << " | (wBE) " << this->currentWBE << " | (wLE) " << this->currentWLE <<
             " | (wJac) " << this->currentWJac << " | (wLan) " << this->currentWLand);
    NR_FUNC_CALLED();

    // Store the global objective function value
    return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac;
}
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::InitialiseSimilarity() {
    F3dContent& con = dynamic_cast<F3dContent&>(*this->con);

    if (this->measure_nmi)
        this->measure->Initialise(*this->measure_nmi, con, conBw.get());

    if (this->measure_ssd)
        this->measure->Initialise(*this->measure_ssd, con, conBw.get());

    if (this->measure_kld)
        this->measure->Initialise(*this->measure_kld, con, conBw.get());

    if (this->measure_lncc)
        this->measure->Initialise(*this->measure_lncc, con, conBw.get());

    if (this->measure_dti)
        this->measure->Initialise(*this->measure_dti, con, conBw.get());

    if (this->measure_mind)
        this->measure->Initialise(*this->measure_mind, con, conBw.get());

    if (this->measure_mindssc)
        this->measure->Initialise(*this->measure_mindssc, con, conBw.get());

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
NiftiImage reg_f3d2<T>::GetBackwardControlPointPositionImage() {
    NR_FUNC_CALLED();
    return controlPointGridBw;
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
        reg_createSymmetricControlPointGrids<T>(this->controlPointGrid,
                                                controlPointGridBw,
                                                this->referencePyramid[0],
                                                this->floatingPyramid[0],
                                                this->affineTransformation.get(),
                                                gridSpacing);
    } else {
        // The control point grid image is initialised with the provided grid
        this->controlPointGrid = this->inputControlPointGrid;
        // The final grid spacing is computed
        this->spacing[0] = this->controlPointGrid->dx / powf(2, this->levelNumber - 1);
        this->spacing[1] = this->controlPointGrid->dy / powf(2, this->levelNumber - 1);
        if (this->controlPointGrid->nz > 1)
            this->spacing[2] = this->controlPointGrid->dz / powf(2, this->levelNumber - 1);
        // The backward grid is derived from the forward
        controlPointGridBw = this->controlPointGrid;
        reg_getDisplacementFromDeformation(controlPointGridBw);
        reg_tools_multiplyValueToImage(controlPointGridBw, controlPointGridBw, -1);
        reg_getDeformationFromDisplacement(controlPointGridBw);
        for (int i = 0; i < controlPointGridBw->num_ext; ++i) {
            mat44 tempMatrix = nifti_mat44_inverse(*reinterpret_cast<mat44 *>(controlPointGridBw->ext_list[i].edata));
            memcpy(controlPointGridBw->ext_list[i].edata, &tempMatrix, sizeof(mat44));
        }
    }

    // Set the floating mask image pyramid
    const unsigned imageCount = this->usePyramid ? this->levelToPerform : 1;
    const unsigned levelCount = this->usePyramid ? this->levelNumber : 1;
    floatingMaskPyramid = vector<unique_ptr<int[]>>(imageCount);

    if (floatingMaskImage)
        reg_createMaskPyramid<T>(floatingMaskImage, floatingMaskPyramid, levelCount, imageCount);
    else
        for (unsigned l = 0; l < imageCount; ++l)
            floatingMaskPyramid[l].reset(new int[this->floatingPyramid[l].nVoxelsPerVolume()]());

    if (inverseConsistencyWeight > 0)
        NR_VERBOSE("Inverse consistency error penalty term weight: "s + std::to_string(inverseConsistencyWeight));

    // Convert the control point grid into velocity field parametrisation
    this->controlPointGrid->intent_p1 = SPLINE_VEL_GRID;
    controlPointGridBw->intent_p1 = SPLINE_VEL_GRID;
    // Set the number of composition to 6 by default
    this->controlPointGrid->intent_p2 = controlPointGridBw->intent_p2 = 6;

    if (this->affineTransformation)
        affineTransformationBw.reset(new mat44(nifti_mat44_inverse(*this->affineTransformation)));

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::ExponentiateGradient() {
    if (!useGradientCumulativeExp) return;

    // Exponentiate the forward gradient using the backward transformation
    NR_DEBUG("Update the forward measure gradient using a Dartel like approach");
    this->compute->ExponentiateGradient(*conBw);

    /* Exponentiate the backward gradient using the forward transformation */
    NR_DEBUG("Update the backward measure gradient using a Dartel like approach");
    computeBw->ExponentiateGradient(*this->con);

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UpdateParameters(float scale) {
    // Restore the last successful control point grids
    this->optimiser->RestoreBestDof();

    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if (bchUpdate) {
        // Forward update
        NR_WARN("USING BCH FORWARD - TESTING ONLY");
        NR_DEBUG("Update the forward control point grid using BCH approximation");
        this->compute->BchUpdate(scale, bchUpdateValue);

        // Backward update
        NR_WARN("USING BCH BACKWARD - TESTING ONLY");
        NR_DEBUG("Update the backward control point grid using BCH approximation");
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
vector<NiftiImage> reg_f3d2<T>::GetWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !this->controlPointGrid || !controlPointGridBw)
        NR_FATAL_ERROR("The reference, floating and control point grid images have to be defined");

    InitCurrentLevel(-1);

    WarpFloatingImage(3); // cubic spline interpolation

    F3dContent& con = dynamic_cast<F3dContent&>(*this->con);
    vector<NiftiImage> warpedImage{
        NiftiImage(con.GetWarped(), NiftiImage::Copy::Image),
        NiftiImage(conBw->GetWarped(), NiftiImage::Copy::Image)
    };

    DeinitCurrentLevel(-1);

    NR_FUNC_CALLED();
    return warpedImage;
}
/* *************************************************************** */
template class reg_f3d2<float>;
