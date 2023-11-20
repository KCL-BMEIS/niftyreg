/**
 *  _reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d.h"
#include "F3dContent.h"

/* *************************************************************** */
template<class T>
reg_f3d<T>::reg_f3d(int refTimePoints, int floTimePoints):
    reg_base<T>::reg_base(refTimePoints, floTimePoints) {

    this->executableName = (char*)"NiftyReg F3D";
    bendingEnergyWeight = 0.001;
    linearEnergyWeight = 0.01;
    jacobianLogWeight = 0;
    jacobianLogApproximation = true;
    spacing[0] = -5;
    spacing[1] = std::numeric_limits<T>::quiet_NaN();
    spacing[2] = std::numeric_limits<T>::quiet_NaN();
    this->useConjGradient = true;
    this->useApproxGradient = false;
    gridRefinement = true;

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetControlPointGridImage(NiftiImage inputControlPointGridIn) {
    inputControlPointGrid = inputControlPointGridIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetBendingEnergyWeight(T be) {
    bendingEnergyWeight = be;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetLinearEnergyWeight(T le) {
    linearEnergyWeight = le;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetJacobianLogWeight(T j) {
    jacobianLogWeight = j;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::ApproximateJacobianLog() {
    jacobianLogApproximation = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DoNotApproximateJacobianLog() {
    jacobianLogApproximation = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetSpacing(unsigned i, T s) {
    spacing[i] = s;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::InitContent(nifti_image *reference, nifti_image *floating, int *mask) {
    unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(this->platform->CreateContentCreator(ContentType::F3d)) };
    this->con.reset(contentCreator->Create(reference, floating, controlPointGrid, this->localWeightSimInput, mask, this->affineTransformation.get(), sizeof(T)));
    this->compute.reset(this->platform->CreateCompute(*this->con));
}
/* *************************************************************** */
template<class T>
T reg_f3d<T>::InitCurrentLevel(int currentLevel) {
    // Set the current input images
    nifti_image *reference, *floating;
    int *mask;
    if (currentLevel < 0) {
        // Settings for GetWarpedImage()
        // Use CPU for warping since CUDA isn't supporting Cubic interpolation
        // TODO Remove this when CUDA supports Cubic interpolation
        this->SetPlatformType(PlatformType::Cpu);
        reference = this->inputReference;
        floating = this->inputFloating;
        mask = nullptr;
    } else {
        const int index = this->usePyramid ? currentLevel : 0;
        reference = this->referencePyramid[index];
        floating = this->floatingPyramid[index];
        mask = this->maskPyramid[index].get();
    }

    // Set the initial step size for the gradient ascent
    T maxStepSize = reference->dx > reference->dy ? reference->dx : reference->dy;
    if (reference->ndim > 2)
        maxStepSize = reference->dz > maxStepSize ? reference->dz : maxStepSize;

    // Refine the control point grid if required
    if (gridRefinement) {
        if (currentLevel == 0) {
            bendingEnergyWeight = bendingEnergyWeight / static_cast<T>(powf(16, this->levelNumber - 1));
            linearEnergyWeight = linearEnergyWeight / static_cast<T>(powf(3, this->levelNumber - 1));
        } else {
            bendingEnergyWeight = bendingEnergyWeight * 16;
            linearEnergyWeight = linearEnergyWeight * 3;
            reg_spline_refineControlPointGrid(controlPointGrid, reference);
        }
    }

    InitContent(reference, floating, mask);

    NR_FUNC_CALLED();
    return maxStepSize;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DeinitCurrentLevel(int currentLevel) {
    reg_base<T>::DeinitCurrentLevel(currentLevel);
    this->compute = nullptr;
    this->con = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CheckParameters() {
    reg_base<T>::CheckParameters();

    // Normalise the objective function weights
    if (strcmp(this->executableName, "NiftyReg F3D") == 0) {
        T penaltySum = bendingEnergyWeight + linearEnergyWeight + jacobianLogWeight + this->landmarkRegWeight;
        if (penaltySum >= 1) {
            this->similarityWeight = 0;
            bendingEnergyWeight /= penaltySum;
            linearEnergyWeight /= penaltySum;
            jacobianLogWeight /= penaltySum;
            this->landmarkRegWeight /= penaltySum;
        } else this->similarityWeight = 1 - penaltySum;
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::Initialise() {
    reg_base<T>::Initialise();

    // Determine the grid spacing and create the grid
    if (!inputControlPointGrid) {
        // Set the spacing along y and z if undefined. Their values are set to match
        // the spacing along the x axis
        if (spacing[1] != spacing[1]) spacing[1] = spacing[0];
        if (spacing[2] != spacing[2]) spacing[2] = spacing[0];

        /* Convert the spacing from voxel to mm if necessary */
        float spacingInMillimetre[3] = {spacing[0], spacing[1], spacing[2]};
        if (spacingInMillimetre[0] < 0) spacingInMillimetre[0] *= -this->inputReference->dx;
        if (spacingInMillimetre[1] < 0) spacingInMillimetre[1] *= -this->inputReference->dy;
        if (spacingInMillimetre[2] < 0) spacingInMillimetre[2] *= -this->inputReference->dz;

        // Define the spacing for the first level
        float gridSpacing[3];
        gridSpacing[0] = spacingInMillimetre[0] * powf(2, this->levelNumber - 1);
        gridSpacing[1] = spacingInMillimetre[1] * powf(2, this->levelNumber - 1);
        gridSpacing[2] = 1;
        if (this->referencePyramid[0]->nz > 1)
            gridSpacing[2] = spacingInMillimetre[2] * powf(2, this->levelNumber - 1);

        // Create and allocate the control point image - by default the transformation is initialised
        // to an identity transformation
        reg_createControlPointGrid<T>(controlPointGrid, this->referencePyramid[0], gridSpacing);

        // The control point grid is updated with an identity transformation
        if (this->affineTransformation)
            reg_affine_getDeformationField(this->affineTransformation.get(), controlPointGrid);
    } else {
        // The control point grid image is initialised with the provided grid
        controlPointGrid = inputControlPointGrid;
        // The final grid spacing is computed
        spacing[0] = controlPointGrid->dx / powf(2, this->levelNumber - 1);
        spacing[1] = controlPointGrid->dy / powf(2, this->levelNumber - 1);
        if (controlPointGrid->nz > 1)
            spacing[2] = controlPointGrid->dz / powf(2, this->levelNumber - 1);
    }

    // Print out some global information about the registration
    NR_VERBOSE("***********************************************************");
    NR_VERBOSE("INPUT PARAMETERS");
    NR_VERBOSE("***********************************************************");
    NR_VERBOSE("Reference image:");
    NR_VERBOSE("\t* name: " << this->inputReference->fname);
    NR_VERBOSE("\t* image dimension: " << this->inputReference->nx << " x " << this->inputReference->ny << " x " <<
               this->inputReference->nz << " x " << this->inputReference->nt);
    NR_VERBOSE("\t* image spacing: " << this->inputReference->dx << " x " << this->inputReference->dy << " x " <<
               this->inputReference->dz << " mm");
    for (int i = 0; i < this->inputReference->nt; i++) {
        NR_VERBOSE("\t* intensity threshold for time point " << i << "/" << this->inputReference->nt - 1 << ": [" <<
                   this->referenceThresholdLow[i] << " " << this->referenceThresholdUp[i] << "]");
        if (this->measure_nmi) {
            if (this->measure_nmi->GetTimePointWeights()[i] > 0) {
                NR_VERBOSE("\t* binning size for time point " << i << "/" << this->inputReference->nt - 1 << ": " <<
                           this->measure_nmi->GetReferenceBinNumber()[i] - 4);
            }
        }
    }
    NR_VERBOSE("\t* gaussian smoothing sigma: " << this->referenceSmoothingSigma);
    NR_VERBOSE("");
    NR_VERBOSE("Floating image:");
    NR_VERBOSE("\t* name: " << this->inputFloating->fname);
    NR_VERBOSE("\t* image dimension: " << this->inputFloating->nx << " x " << this->inputFloating->ny << " x " <<
               this->inputFloating->nz << " x " << this->inputFloating->nt);
    NR_VERBOSE("\t* image spacing: " << this->inputFloating->dx << " x " << this->inputFloating->dy << " x " <<
               this->inputFloating->dz << " mm");
    for (int i = 0; i < this->inputFloating->nt; i++) {
        NR_VERBOSE("\t* intensity threshold for time point " << i << "/" << this->inputFloating->nt - 1 << ": [" <<
                   this->floatingThresholdLow[i] << " " << this->floatingThresholdUp[i] << "]");
        if (this->measure_nmi) {
            if (this->measure_nmi->GetTimePointWeights()[i] > 0) {
                NR_VERBOSE("\t* binning size for time point " << i << "/" << this->inputFloating->nt - 1 << ": " <<
                           this->measure_nmi->GetFloatingBinNumber()[i] - 4);
            }
        }
    }
    NR_VERBOSE("\t* gaussian smoothing sigma: " << this->floatingSmoothingSigma);
    NR_VERBOSE("");
    NR_VERBOSE("Warped image padding value: " << this->warpedPaddingValue);
    NR_VERBOSE("");
    NR_VERBOSE("Level number: " << this->levelNumber);
    if (this->levelNumber != this->levelToPerform)
        NR_VERBOSE("\t* Level to perform: " << this->levelToPerform);
    NR_VERBOSE("");
    NR_VERBOSE("Maximum iteration number during the last level: " << this->maxIterationNumber);
    NR_VERBOSE("");

    NR_VERBOSE("Final spacing in mm: " << spacing[0] << " " << spacing[1] << " " << spacing[2]);
    NR_VERBOSE("");
    if (this->measure_ssd)
        NR_VERBOSE("The SSD is used as a similarity measure.");
    if (this->measure_kld)
        NR_VERBOSE("The KL divergence is used as a similarity measure.");
    if (this->measure_lncc)
        NR_VERBOSE("The LNCC is used as a similarity measure.");
    if (this->measure_dti)
        NR_VERBOSE("A DTI based measure is used as a similarity measure.");
    if (this->measure_mind)
        NR_VERBOSE("MIND is used as a similarity measure.");
    if (this->measure_mindssc)
        NR_VERBOSE("MINDSSC is used as a similarity measure.");
    if (this->measure_nmi || (!this->measure_dti && !this->measure_kld && !this->measure_lncc &&
                              !this->measure_nmi && !this->measure_ssd && !this->measure_mind && !this->measure_mindssc))
        NR_VERBOSE("The NMI is used as a similarity measure.");
    NR_VERBOSE("Similarity measure term weight: " << this->similarityWeight);
    NR_VERBOSE("");
    if (bendingEnergyWeight > 0) {
        NR_VERBOSE("Bending energy penalty term weight: " << bendingEnergyWeight);
        NR_VERBOSE("");
    }
    if (linearEnergyWeight > 0) {
        NR_VERBOSE("Linear energy penalty term weight: " << linearEnergyWeight);
        NR_VERBOSE("");
    }
    if (jacobianLogWeight > 0) {
        NR_VERBOSE("Jacobian-based penalty term weight: " << jacobianLogWeight);
        NR_VERBOSE("\t* Jacobian-based penalty term is " << (jacobianLogApproximation ? "approximated" : "not approximated"));
        NR_VERBOSE("");
    }
    if (this->landmarkRegWeight > 0) {
        NR_VERBOSE("Landmark distance regularisation term weight: " << this->landmarkRegWeight);
        NR_VERBOSE("");
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetDeformationField() {
    this->compute->GetDeformationField(false, // Composition
                                       true); // bspline
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type) {
    if (jacobianLogWeight <= 0) return 0;

    bool approx = type == 2 ? false : jacobianLogApproximation;

    double value = this->compute->GetJacobianPenaltyTerm(approx);

    unsigned maxit = 5;
    if (type > 0) maxit = 20;
    unsigned it = 0;
    while (value != value && it < maxit) {
        value = this->compute->CorrectFolding(approx);
        NR_DEBUG("Folding correction");
        it++;
    }
    if (type > 0) {
        if (value != value) {
            this->optimiser->RestoreBestDof();
            NR_WARN_WFCT("The folding correction scheme failed");
        } else if (it > 0) {
            NR_DEBUG("Folding correction, " << it << " step(s)");
        }
    }
    NR_FUNC_CALLED();
    return jacobianLogWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm() {
    if (bendingEnergyWeight <= 0) return 0;
    const double value = this->compute->ApproxBendingEnergy();
    NR_FUNC_CALLED();
    return bendingEnergyWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm() {
    if (linearEnergyWeight <= 0) return 0;
    const double value = this->compute->ApproxLinearEnergy();
    NR_FUNC_CALLED();
    return linearEnergyWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm() {
    if (this->landmarkRegWeight <= 0) return 0;
    const double value = this->compute->GetLandmarkDistance(this->landmarkRegNumber,
                                                            this->landmarkReference,
                                                            this->landmarkFloating);
    NR_FUNC_CALLED();
    return this->landmarkRegWeight * value;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetSimilarityMeasureGradient() {
    this->GetVoxelBasedGradient();

    // The voxel-based NMI gradient is convolved with a spline kernel
    // And the node-based NMI gradient is extracted
    this->compute->ConvolveVoxelBasedMeasureGradient(this->similarityWeight);

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetBendingEnergyGradient() {
    if (bendingEnergyWeight <= 0) return;
    this->compute->ApproxBendingEnergyGradient(bendingEnergyWeight);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetLinearEnergyGradient() {
    if (linearEnergyWeight <= 0) return;
    this->compute->ApproxLinearEnergyGradient(linearEnergyWeight);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetJacobianBasedGradient() {
    if (jacobianLogWeight <= 0) return;
    this->compute->JacobianPenaltyTermGradient(jacobianLogWeight, jacobianLogApproximation);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetLandmarkDistanceGradient() {
    if (this->landmarkRegWeight <= 0) return;
    this->compute->LandmarkDistanceGradient(this->landmarkRegNumber,
                                            this->landmarkReference,
                                            this->landmarkFloating,
                                            this->landmarkRegWeight);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
T reg_f3d<T>::NormaliseGradient() {
    // First compute the gradient max length for normalisation purpose
    T maxGradLength = (T)this->compute->GetMaximalLength(this->optimiseX, this->optimiseY, this->optimiseZ);

    if (strcmp(this->executableName, "NiftyReg F3D") == 0) {
        // The gradient is normalised if we are running f3d
        // It will be normalised later when running f3d2
        this->compute->NormaliseGradient(maxGradLength, this->optimiseX, this->optimiseY, this->optimiseZ);
        NR_DEBUG("Objective function gradient maximal length: " << maxGradLength);
    }

    NR_FUNC_CALLED();

    // Returns the largest gradient distance
    return maxGradLength;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DisplayCurrentLevelParameters(int currentLevel) {
    const nifti_image *reference = this->con->Content::GetReference();
    const nifti_image *floating = this->con->Content::GetFloating();
    NR_VERBOSE("Current level: " << currentLevel + 1 << " / " << this->levelNumber);
    NR_VERBOSE("Maximum iteration number: " << this->maxIterationNumber);
    NR_VERBOSE("Current reference image");
    NR_VERBOSE("\t* image dimension: " << reference->nx << " x " << reference->ny << " x " << reference->nz << " x " << reference->nt);
    NR_VERBOSE("\t* image spacing: " << reference->dx << " x " << reference->dy << " x " << reference->dz << " mm");
    NR_VERBOSE("Current floating image");
    NR_VERBOSE("\t* image dimension: " << floating->nx << " x " << floating->ny << " x " << floating->nz << " x " << floating->nt);
    NR_VERBOSE("\t* image spacing: " << floating->dx << " x " << floating->dy << " x " << floating->dz << " mm");
    NR_VERBOSE("Current control point image");
    NR_VERBOSE("\t* image dimension: " << controlPointGrid->nx << " x " << controlPointGrid->ny << " x " << controlPointGrid->nz);
    NR_VERBOSE("\t* image spacing: " << controlPointGrid->dx << " x " << controlPointGrid->dy << " x " << controlPointGrid->dz << " mm");

    // Input matrices are only printed out in debug
    if (reference->sform_code > 0)
        NR_MAT44_DEBUG(reference->sto_xyz, "Reference sform");
    else NR_MAT44_DEBUG(reference->qto_xyz, "Reference qform");
    if (floating->sform_code > 0)
        NR_MAT44_DEBUG(floating->sto_xyz, "Floating sform");
    else NR_MAT44_DEBUG(floating->qto_xyz, "Floating qform");
    if (controlPointGrid->sform_code > 0)
        NR_MAT44_DEBUG(controlPointGrid->sto_xyz, "CPP sform");
    else NR_MAT44_DEBUG(controlPointGrid->qto_xyz, "CPP qform");

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::GetObjectiveFunctionValue() {
    currentWJac = ComputeJacobianBasedPenaltyTerm(1); // 20 iterations
    currentWBE = ComputeBendingEnergyPenaltyTerm();
    currentWLE = ComputeLinearEnergyPenaltyTerm();
    this->currentWLand = ComputeLandmarkDistancePenaltyTerm();

    // Compute initial similarity measure
    this->currentWMeasure = 0;
    if (this->similarityWeight > 0) {
        this->WarpFloatingImage(this->interpolation);
        this->currentWMeasure = this->ComputeSimilarityMeasure();
    }

    NR_DEBUG("(wMeasure) " << this->currentWMeasure << " | (wBE) " << currentWBE << " | (wLE) " << currentWLE <<
             " | (wJac) " << currentWJac << " | (wLan) " << this->currentWLand);
    NR_FUNC_CALLED();

    // Store the global objective function value
    return this->currentWMeasure - currentWBE - currentWLE - currentWJac - this->currentWLand;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateParameters(float scale) {
    this->compute->UpdateControlPointPosition(this->optimiser->GetCurrentDof(),
                                              this->optimiser->GetBestDof(),
                                              this->optimiser->GetGradient(),
                                              scale,
                                              this->optimiseX,
                                              this->optimiseY,
                                              this->optimiseZ);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetOptimiser() {
    this->optimiser.reset(this->platform->template CreateOptimiser<T>(dynamic_cast<F3dContent&>(*this->con),
                                                                      *this,
                                                                      this->maxIterationNumber,
                                                                      this->useConjGradient,
                                                                      this->optimiseX,
                                                                      this->optimiseY,
                                                                      this->optimiseZ));
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SmoothGradient() {
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (this->gradientSmoothingSigma == 0) return;
    this->compute->SmoothGradient(this->gradientSmoothingSigma);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetApproximatedGradient() {
    this->compute->GetApproximatedGradient(*this);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
vector<NiftiImage> reg_f3d<T>::GetWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !controlPointGrid)
        NR_FATAL_ERROR("The reference, floating and control point grid images have to be defined");

    InitCurrentLevel(-1);
    this->WarpFloatingImage(3); // cubic spline interpolation
    NiftiImage warpedImage = NiftiImage(this->con->GetWarped(), NiftiImage::Copy::Image);
    DeinitCurrentLevel(-1);

    NR_FUNC_CALLED();
    return { warpedImage };
}
/* *************************************************************** */
template<class T>
NiftiImage reg_f3d<T>::GetControlPointPositionImage() {
    NR_FUNC_CALLED();
    return controlPointGrid;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateBestObjFunctionValue() {
    this->bestWMeasure = this->currentWMeasure;
    bestWBE = currentWBE;
    bestWLE = currentWLE;
    bestWJac = currentWJac;
    this->bestWLand = this->currentWLand;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintInitialObjFunctionValue() {
    NR_VERBOSE("Initial objective function: " << this->optimiser->GetBestObjFunctionValue() << " = (wSIM)" << this->bestWMeasure <<
               " - (wBE)" << bestWBE << " - (wLE)" << bestWLE << " - (wJAC)" << bestWJac << " - (wLAN)" << this->bestWLand);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintCurrentObjFunctionValue(T currentSize) {
    NR_VERBOSE("[" << this->optimiser->GetCurrentIterationNumber() << "] Current objective function: " <<
               this->optimiser->GetBestObjFunctionValue() << " = (wSIM)" << this->bestWMeasure <<
               (bendingEnergyWeight > 0 ? " - (wBE)"s + std::to_string(bestWBE) : "") <<
               (linearEnergyWeight > 0 ? " - (wLE)"s + std::to_string(bestWLE) : "") <<
               (jacobianLogWeight > 0 ? " - (wJAC)"s + std::to_string(bestWJac) : "") <<
               (this->landmarkRegWeight > 0 ? " - (wLAN)"s + std::to_string(this->bestWLand) : "") <<
               " [+ " << currentSize << " mm]");
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetObjectiveFunctionGradient() {
    if (!this->useApproxGradient) {
        // Compute the gradient of the similarity measure
        if (this->similarityWeight > 0) {
            this->WarpFloatingImage(this->interpolation);
            GetSimilarityMeasureGradient();
        } else {
            dynamic_cast<F3dContent&>(*this->con).ZeroTransformationGradient();
        }
        // Compute the penalty term gradients if required
        GetBendingEnergyGradient();
        GetJacobianBasedGradient();
        GetLinearEnergyGradient();
        GetLandmarkDistanceGradient();
    } else {
        GetApproximatedGradient();
    }

    this->optimiser->IncrementCurrentIterationNumber();

    // Smooth the gradient if require
    SmoothGradient();
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CorrectTransformation() {
    if (jacobianLogWeight > 0 && jacobianLogApproximation)
        ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template class reg_f3d<float>;
