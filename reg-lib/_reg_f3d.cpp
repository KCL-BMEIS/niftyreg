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

#ifdef _USE_CUDA
#include "CudaF3dContent.h"
#endif

 /* *************************************************************** */
template<class T>
reg_f3d<T>::reg_f3d(int refTimePoint, int floTimePoint):
    reg_base<T>::reg_base(refTimePoint, floTimePoint) {

    this->executableName = (char*)"NiftyReg F3D";
    inputControlPointGrid = nullptr; // pointer to external
    controlPointGrid = nullptr;
    bendingEnergyWeight = 0.001;
    linearEnergyWeight = 0.00;
    jacobianLogWeight = 0;
    jacobianLogApproximation = true;
    spacing[0] = -5;
    spacing[1] = std::numeric_limits<T>::quiet_NaN();
    spacing[2] = std::numeric_limits<T>::quiet_NaN();
    this->useConjGradient = true;
    this->useApproxGradient = false;
    gridRefinement = true;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::reg_f3d");
#endif
}
/* *************************************************************** */
template<class T>
reg_f3d<T>::~reg_f3d() {
    if (controlPointGrid) {
        nifti_image_free(controlPointGrid);
        controlPointGrid = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::~reg_f3d");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetControlPointGridImage(nifti_image *cp) {
    inputControlPointGrid = cp;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetControlPointGridImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetBendingEnergyWeight(T be) {
    bendingEnergyWeight = be;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetBendingEnergyWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetLinearEnergyWeight(T le) {
    linearEnergyWeight = le;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetLinearEnergyWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetJacobianLogWeight(T j) {
    jacobianLogWeight = j;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetJacobianLogWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::ApproximateJacobianLog() {
    jacobianLogApproximation = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ApproximateJacobianLog");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DoNotApproximateJacobianLog() {
    jacobianLogApproximation = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::DoNotApproximateJacobianLog");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetSpacing(unsigned int i, T s) {
    spacing[i] = s;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetSpacing");
#endif
}
/* *************************************************************** */
template<class T>
T reg_f3d<T>::InitialiseCurrentLevel(nifti_image *reference) {
    // Set the initial step size for the gradient ascent
    T maxStepSize = reference->dx > reference->dy ? reference->dx : reference->dy;
    if (reference->ndim > 2)
        maxStepSize = reference->dz > maxStepSize ? reference->dz : maxStepSize;

    // Refine the control point grid if required
    if (gridRefinement) {
        if (this->currentLevel == 0) {
            bendingEnergyWeight = bendingEnergyWeight / static_cast<T>(powf(16, this->levelNumber - 1));
            linearEnergyWeight = linearEnergyWeight / static_cast<T>(powf(3, this->levelNumber - 1));
        } else {
            bendingEnergyWeight = bendingEnergyWeight * static_cast<T>(16);
            linearEnergyWeight = linearEnergyWeight * static_cast<T>(3);
            reg_spline_refineControlPointGrid(controlPointGrid, reference);
        }
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::InitialiseCurrentLevel");
#endif
    return maxStepSize;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CheckParameters() {
    reg_base<T>::CheckParameters();
    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
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
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::CheckParameters");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::Initialise() {
    if (this->initialised) return;

    reg_base<T>::Initialise();

    // DETERMINE THE GRID SPACING AND CREATE THE GRID
    if (!inputControlPointGrid) {
        // Set the spacing along y and z if undefined. Their values are set to match
        // the spacing along the x axis
        if (spacing[1] != spacing[1]) spacing[1] = spacing[0];
        if (spacing[2] != spacing[2]) spacing[2] = spacing[0];

        /* Convert the spacing from voxel to mm if necessary */
        float spacingInMillimeter[3] = {spacing[0], spacing[1], spacing[2]};
        if (spacingInMillimeter[0] < 0) spacingInMillimeter[0] *= -this->inputReference->dx;
        if (spacingInMillimeter[1] < 0) spacingInMillimeter[1] *= -this->inputReference->dy;
        if (spacingInMillimeter[2] < 0) spacingInMillimeter[2] *= -this->inputReference->dz;

        // Define the spacing for the first level
        float gridSpacing[3];
        gridSpacing[0] = spacingInMillimeter[0] * powf(2, this->levelNumber - 1);
        gridSpacing[1] = spacingInMillimeter[1] * powf(2, this->levelNumber - 1);
        gridSpacing[2] = 1;
        if (this->referencePyramid[0]->nz > 1)
            gridSpacing[2] = spacingInMillimeter[2] * powf(2, this->levelNumber - 1);

        // Create and allocate the control point image
        reg_createControlPointGrid<T>(&controlPointGrid, this->referencePyramid[0], gridSpacing);

        // The control point position image is initialised with the affine transformation
        if (!this->affineTransformation) {
            memset(controlPointGrid->data, 0, controlPointGrid->nvox * controlPointGrid->nbyper);
            reg_tools_multiplyValueToImage(controlPointGrid, controlPointGrid, 0.f);
            reg_getDeformationFromDisplacement(controlPointGrid);
        } else reg_affine_getDeformationField(this->affineTransformation, controlPointGrid);
    } else {
        // The control point grid image is initialised with the provided grid
        controlPointGrid = nifti_copy_nim_info(inputControlPointGrid);
        controlPointGrid->data = malloc(controlPointGrid->nvox * controlPointGrid->nbyper);
        memcpy(controlPointGrid->data, inputControlPointGrid->data,
               controlPointGrid->nvox * controlPointGrid->nbyper);
        // The final grid spacing is computed
        spacing[0] = controlPointGrid->dx / powf(2, this->levelNumber - 1);
        spacing[1] = controlPointGrid->dy / powf(2, this->levelNumber - 1);
        if (controlPointGrid->nz > 1)
            spacing[2] = controlPointGrid->dz / powf(2, this->levelNumber - 1);
    }
#ifdef NDEBUG
    if (this->verbose) {
#endif
        std::string text;
        // Print out some global information about the registration
        reg_print_info(this->executableName, "***********************************************************");
        reg_print_info(this->executableName, "INPUT PARAMETERS");
        reg_print_info(this->executableName, "***********************************************************");
        reg_print_info(this->executableName, "Reference image:");
        text = stringFormat("\t* name: %s", this->inputReference->fname);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t* image dimension: %i x %i x %i x %i",
                            this->inputReference->nx, this->inputReference->ny,
                            this->inputReference->nz, this->inputReference->nt);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t* image spacing: %g x %g x %g mm",
                            this->inputReference->dx, this->inputReference->dy, this->inputReference->dz);
        reg_print_info(this->executableName, text.c_str());
        for (int i = 0; i < this->inputReference->nt; i++) {
            text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                                i, this->inputReference->nt - 1, this->referenceThresholdLow[i], this->referenceThresholdUp[i]);
            reg_print_info(this->executableName, text.c_str());
            if (this->measure_nmi) {
                if (this->measure_nmi->GetTimepointsWeights()[i] > 0) {
                    text = stringFormat("\t* binning size for timepoint %i/%i: %i",
                                        i, this->inputFloating->nt - 1, this->measure_nmi->GetReferenceBinNumber()[i] - 4);
                    reg_print_info(this->executableName, text.c_str());
                }
            }
        }
        text = stringFormat("\t* gaussian smoothing sigma: %g", this->referenceSmoothingSigma);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");
        reg_print_info(this->executableName, "Floating image:");
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t* name: %s", this->inputFloating->fname);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t* image dimension: %i x %i x %i x %i",
                            this->inputFloating->nx, this->inputFloating->ny,
                            this->inputFloating->nz, this->inputFloating->nt);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t* image spacing: %g x %g x %g mm", this->inputFloating->dx,
                            this->inputFloating->dy, this->inputFloating->dz);
        reg_print_info(this->executableName, text.c_str());
        for (int i = 0; i < this->inputFloating->nt; i++) {
            text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                                i, this->inputFloating->nt - 1, this->floatingThresholdLow[i], this->floatingThresholdUp[i]);
            reg_print_info(this->executableName, text.c_str());
            if (this->measure_nmi) {
                if (this->measure_nmi->GetTimepointsWeights()[i] > 0) {
                    text = stringFormat("\t* binning size for timepoint %i/%i: %i",
                                        i, this->inputFloating->nt - 1, this->measure_nmi->GetFloatingBinNumber()[i] - 4);
                    reg_print_info(this->executableName, text.c_str());
                }
            }
        }
        text = stringFormat("\t* gaussian smoothing sigma: %g", this->floatingSmoothingSigma);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");
        text = stringFormat("Warped image padding value: %g", this->warpedPaddingValue);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");
        text = stringFormat("Level number: %i", this->levelNumber);
        reg_print_info(this->executableName, text.c_str());
        if (this->levelNumber != this->levelToPerform) {
            text = stringFormat("\t* Level to perform: %i", this->levelToPerform);
            reg_print_info(this->executableName, text.c_str());
        }
        reg_print_info(this->executableName, "");
        text = stringFormat("Maximum iteration number during the last level: %i", (int)this->maxIterationNumber);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");

        text = stringFormat("Final spacing in mm: %g %g %g", spacing[0], spacing[1], spacing[2]);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");
        if (this->measure_ssd)
            reg_print_info(this->executableName, "The SSD is used as a similarity measure.");
        if (this->measure_kld)
            reg_print_info(this->executableName, "The KL divergence is used as a similarity measure.");
        if (this->measure_lncc)
            reg_print_info(this->executableName, "The LNCC is used as a similarity measure.");
        if (this->measure_dti)
            reg_print_info(this->executableName, "A DTI based measure is used as a similarity measure.");
        if (this->measure_mind)
            reg_print_info(this->executableName, "MIND is used as a similarity measure.");
        if (this->measure_mindssc)
            reg_print_info(this->executableName, "MINDSSC is used as a similarity measure.");
        if (this->measure_nmi || (!this->measure_dti && !this->measure_kld && !this->measure_lncc &&
                                  !this->measure_nmi && !this->measure_ssd && !this->measure_mind && !this->measure_mindssc))
            reg_print_info(this->executableName, "The NMI is used as a similarity measure.");
        text = stringFormat("Similarity measure term weight: %g", this->similarityWeight);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "");
        if (bendingEnergyWeight > 0) {
            text = stringFormat("Bending energy penalty term weight: %g", bendingEnergyWeight);
            reg_print_info(this->executableName, text.c_str());
            reg_print_info(this->executableName, "");
        }
        if ((linearEnergyWeight) > 0) {
            text = stringFormat("Linear energy penalty term weight: %g", linearEnergyWeight);
            reg_print_info(this->executableName, text.c_str());
            reg_print_info(this->executableName, "");
        }
        if (jacobianLogWeight > 0) {
            text = stringFormat("Jacobian-based penalty term weight: %g", jacobianLogWeight);
            reg_print_info(this->executableName, text.c_str());
            if (jacobianLogApproximation) {
                reg_print_info(this->executableName, "\t* Jacobian-based penalty term is approximated");
            } else {
                reg_print_info(this->executableName, "\t* Jacobian-based penalty term is not approximated");
            }
            reg_print_info(this->executableName, "");
        }
        if (this->landmarkRegWeight > 0) {
            text = stringFormat("Landmark distance regularisation term weight: %g", this->landmarkRegWeight);
            reg_print_info(this->executableName, text.c_str());
            reg_print_info(this->executableName, "");
        }
#ifdef NDEBUG
    }
#endif

    this->initialised = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::Initialise");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::InitContent(nifti_image *reference, nifti_image *floating, int *mask) {
    if (this->platformType == PlatformType::Cpu)
        this->con = new F3dContent(reference, floating, controlPointGrid, this->localWeightSimInput, mask, this->affineTransformation, sizeof(T));
#ifdef _USE_CUDA
    else if (this->platformType == PlatformType::Cuda)
        this->con = new CudaF3dContent(reference, floating, controlPointGrid, this->localWeightSimInput, mask, this->affineTransformation, sizeof(T));
#endif
    this->compute = this->platform->CreateCompute(*this->con);
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DeinitContent() {
    delete this->compute;
    this->compute = nullptr;
    delete this->con;
    this->con = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetDeformationField() {
    this->compute->GetDeformationField(false, // Composition
                                       true); // bspline
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetDeformationField");
#endif
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type) {
    if (jacobianLogWeight <= 0) return 0;

    bool approx = type == 2 ? false : jacobianLogApproximation;

    double value = this->compute->GetJacobianPenaltyTerm(approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (value != value && it < maxit) {
        value = this->compute->CorrectFolding(approx);
#ifndef NDEBUG
        reg_print_msg_debug("Folding correction");
#endif
        it++;
    }
    if (type > 0) {
        if (value != value) {
            this->optimiser->RestoreBestDOF();
            reg_print_fct_warn("reg_f3d<T>::ComputeJacobianBasedPenaltyTerm()");
            reg_print_msg_warn("The folding correction scheme failed");
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
    reg_print_fct_debug("reg_f3d<T>::ComputeJacobianBasedPenaltyTerm");
#endif
    return jacobianLogWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm() {
    if (bendingEnergyWeight <= 0) return 0;

    double value = this->compute->ApproxBendingEnergy();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeBendingEnergyPenaltyTerm");
#endif
    return bendingEnergyWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm() {
    if (linearEnergyWeight <= 0) return 0;

    double value = this->compute->ApproxLinearEnergy();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeLinearEnergyPenaltyTerm");
#endif
    return linearEnergyWeight * value;
}
/* *************************************************************** */
template<class T>
double reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm() {
    if (this->landmarkRegWeight <= 0) return 0;

    double value = this->compute->GetLandmarkDistance(this->landmarkRegNumber,
                                                      this->landmarkReference,
                                                      this->landmarkFloating);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm");
#endif
    return this->landmarkRegWeight * value;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetSimilarityMeasureGradient() {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    this->GetVoxelBasedGradient();

    nifti_image *voxelBasedMeasureGradient = dynamic_cast<F3dContent*>(this->con)->GetVoxelBasedMeasureGradient();
    const int kernel_type = CUBIC_SPLINE_KERNEL;
    // The voxel based NMI gradient is convolved with a spline kernel
    // Convolution along the x axis
    float currentNodeSpacing[3];
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dx;
    bool activeAxis[3] = {1, 0, 0};
    reg_tools_kernelConvolution(voxelBasedMeasureGradient,
                                currentNodeSpacing,
                                kernel_type,
                                nullptr, // mask
                                nullptr, // all volumes are considered as active
                                activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dy;
    activeAxis[0] = 0;
    activeAxis[1] = 1;
    reg_tools_kernelConvolution(voxelBasedMeasureGradient,
                                currentNodeSpacing,
                                kernel_type,
                                nullptr, // mask
                                nullptr, // all volumes are considered as active
                                activeAxis);
    // Convolution along the z axis if required
    if (voxelBasedMeasureGradient->nz > 1) {
        currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dz;
        activeAxis[1] = 0;
        activeAxis[2] = 1;
        reg_tools_kernelConvolution(voxelBasedMeasureGradient,
                                    currentNodeSpacing,
                                    kernel_type,
                                    nullptr, // mask
                                    nullptr, // all volumes are considered as active
                                    activeAxis);
    }

    // Update the changes for GPU
    dynamic_cast<F3dContent*>(this->con)->UpdateVoxelBasedMeasureGradient();

    // The node based NMI gradient is extracted
    this->compute->VoxelCentricToNodeCentric(this->similarityWeight);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetSimilarityMeasureGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetBendingEnergyGradient() {
    if (bendingEnergyWeight <= 0) return;

    this->compute->ApproxBendingEnergyGradient(bendingEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetBendingEnergyGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetLinearEnergyGradient() {
    if (linearEnergyWeight <= 0) return;

    this->compute->ApproxLinearEnergyGradient(linearEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetLinearEnergyGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetJacobianBasedGradient() {
    if (jacobianLogWeight <= 0) return;

    this->compute->JacobianPenaltyTermGradient(jacobianLogWeight, jacobianLogApproximation);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetLandmarkDistanceGradient() {
    if (this->landmarkRegWeight <= 0) return;

    this->compute->LandmarkDistanceGradient(this->landmarkRegNumber,
                                            this->landmarkReference,
                                            this->landmarkFloating,
                                            this->landmarkRegWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetLandmarkDistanceGradient");
#endif
}
/* *************************************************************** */
template<class T>
T reg_f3d<T>::NormaliseGradient() {
    // First compute the gradient max length for normalisation purpose
    T maxGradLength = (T)this->compute->GetMaximalLength(this->optimiser->GetVoxNumber(), this->optimiseX, this->optimiseY, this->optimiseZ);

    if (strcmp(this->executableName, "NiftyReg F3D") == 0) {
        // The gradient is normalised if we are running f3d
        // It will be normalised later when running f3d_sym or f3d2
        this->compute->NormaliseGradient(this->optimiser->GetVoxNumber(), maxGradLength);
#ifndef NDEBUG
        char text[255];
        sprintf(text, "Objective function gradient maximal length: %g", maxGradLength);
        reg_print_msg_debug(text);
#endif
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::NormaliseGradient");
#endif

    // Returns the largest gradient distance
    return maxGradLength;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DisplayCurrentLevelParameters() {
#ifdef NDEBUG
    if (this->verbose) {
#endif
        nifti_image *reference = this->con->Content::GetReference();
        nifti_image *floating = this->con->Content::GetFloating();
        char text[255];
        sprintf(text, "Current level: %i / %i", this->currentLevel + 1, this->levelNumber);
        reg_print_info(this->executableName, text);
        sprintf(text, "Maximum iteration number: %i", (int)this->maxIterationNumber);
        reg_print_info(this->executableName, text);
        reg_print_info(this->executableName, "Current reference image");
        sprintf(text, "\t* image dimension: %i x %i x %i x %i", reference->nx, reference->ny, reference->nz, reference->nt);
        reg_print_info(this->executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm", reference->dx, reference->dy, reference->dz);
        reg_print_info(this->executableName, text);
        reg_print_info(this->executableName, "Current floating image");
        sprintf(text, "\t* image dimension: %i x %i x %i x %i", floating->nx, floating->ny, floating->nz, floating->nt);
        reg_print_info(this->executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm", floating->dx, floating->dy, floating->dz);
        reg_print_info(this->executableName, text);
        reg_print_info(this->executableName, "Current control point image");
        sprintf(text, "\t* image dimension: %i x %i x %i", controlPointGrid->nx, controlPointGrid->ny, controlPointGrid->nz);
        reg_print_info(this->executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm", controlPointGrid->dx, controlPointGrid->dy, controlPointGrid->dz);
        reg_print_info(this->executableName, text);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    if (reference->sform_code > 0)
        reg_mat44_disp(&(reference->sto_xyz), (char *)"[NiftyReg DEBUG] Reference sform");
    else reg_mat44_disp(&(reference->qto_xyz), (char *)"[NiftyReg DEBUG] Reference qform");

    if (floating->sform_code > 0)
        reg_mat44_disp(&(floating->sto_xyz), (char *)"[NiftyReg DEBUG] Floating sform");
    else reg_mat44_disp(&(floating->qto_xyz), (char *)"[NiftyReg DEBUG] Floating qform");

    if (controlPointGrid->sform_code > 0)
        reg_mat44_disp(&(controlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] CPP sform");
    else reg_mat44_disp(&(controlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] CPP qform");
#endif
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::DisplayCurrentLevelParameters");
#endif
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
#ifndef NDEBUG
    char text[255];
    sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g | (wLan) %g",
            this->currentWMeasure, currentWBE, currentWLE, currentWJac, this->currentWLand);
    reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif
    // Store the global objective function value

    return this->currentWMeasure - currentWBE - currentWLE - currentWJac - this->currentWLand;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateParameters(float scale) {
    this->compute->UpdateControlPointPosition(this->optimiser->GetCurrentDOF(),
                                              this->optimiser->GetBestDOF(),
                                              this->optimiser->GetGradient(),
                                              scale,
                                              this->optimiseX,
                                              this->optimiseY,
                                              this->optimiseZ);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::UpdateParameters");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetOptimiser() {
    this->optimiser = this->platform->template CreateOptimiser<T>(*dynamic_cast<F3dContent*>(this->con),
                                                                  *this,
                                                                  this->maxIterationNumber,
                                                                  this->useConjGradient,
                                                                  this->optimiseX,
                                                                  this->optimiseY,
                                                                  this->optimiseZ);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SmoothGradient() {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (this->gradientSmoothingSigma != 0) {
        float kernel = fabs(this->gradientSmoothingSigma);
        F3dContent *con = dynamic_cast<F3dContent*>(this->con);
        reg_tools_kernelConvolution(con->GetTransformationGradient(), &kernel, GAUSSIAN_KERNEL);
        // Update the changes for GPU
        con->UpdateTransformationGradient();
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetApproximatedGradient() {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    F3dContent *con = dynamic_cast<F3dContent*>(this->con);
    nifti_image *controlPointGrid = con->GetControlPointGrid();
    nifti_image *transformationGradient = con->GetTransformationGradient();

    // Loop over every control point
    T *gridPtr = static_cast<T*>(controlPointGrid->data);
    T *gradPtr = static_cast<T*>(transformationGradient->data);
    T eps = controlPointGrid->dx / 100.f;
    for (size_t i = 0; i < controlPointGrid->nvox; ++i) {
        T currentValue = this->optimiser->GetBestDOF()[i];
        gridPtr[i] = currentValue + eps;
        // Update the changes for GPU
        con->UpdateControlPointGrid();
        double valPlus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue - eps;
        // Update the changes for GPU
        con->UpdateControlPointGrid();
        double valMinus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue;
        // Update the changes for GPU
        con->UpdateControlPointGrid();
        gradPtr[i] = -(T)((valPlus - valMinus) / (2.0 * eps));
    }

    // Update the changes for GPU
    con->UpdateTransformationGradient();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
nifti_image** reg_f3d<T>::GetWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !controlPointGrid) {
        reg_print_fct_error("reg_f3d<T>::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    InitialiseCurrentLevel(this->inputReference);
    InitContent(this->inputReference, this->inputFloating, nullptr);

    this->WarpFloatingImage(3); // cubic spline interpolation

    nifti_image **warpedImage = (nifti_image**)calloc(2, sizeof(nifti_image*));
    warpedImage[0] = this->con->GetWarped(0);
    if (this->inputFloating->nt == 2)
        warpedImage[1] = this->con->GetWarped(1);

    this->con->SetWarped(nullptr); // Prevent deallocating of warpedImage
    DeinitContent();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetWarpedImage");
#endif
    return warpedImage;
}
/* *************************************************************** */
template<class T>
nifti_image* reg_f3d<T>::GetControlPointPositionImage() {
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(controlPointGrid);
    returnedControlPointGrid->data = malloc(returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    memcpy(returnedControlPointGrid->data, controlPointGrid->data,
           returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    return returnedControlPointGrid;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetControlPointPositionImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateBestObjFunctionValue() {
    this->bestWMeasure = this->currentWMeasure;
    bestWBE = currentWBE;
    bestWLE = currentWLE;
    bestWJac = currentWJac;
    this->bestWLand = this->currentWLand;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintInitialObjFunctionValue() {
    if (!this->verbose) return;

    double bestValue = this->optimiser->GetBestObjFunctionValue();

    char text[255];
    sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wLAN)%g",
            bestValue, this->bestWMeasure, bestWBE, bestWLE, bestWJac, this->bestWLand);
    reg_print_info(this->executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::PrintInitialObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintCurrentObjFunctionValue(T currentSize) {
    if (!this->verbose) return;

    char text[255];
    sprintf(text, "[%i] Current objective function: %g",
            (int)this->optimiser->GetCurrentIterationNumber(),
            this->optimiser->GetBestObjFunctionValue());
    sprintf(text + strlen(text), " = (wSIM)%g", this->bestWMeasure);
    if (bendingEnergyWeight > 0)
        sprintf(text + strlen(text), " - (wBE)%.2e", bestWBE);
    if (linearEnergyWeight > 0)
        sprintf(text + strlen(text), " - (wLE)%.2e", bestWLE);
    if (jacobianLogWeight > 0)
        sprintf(text + strlen(text), " - (wJAC)%.2e", bestWJac);
    if (this->landmarkRegWeight > 0)
        sprintf(text + strlen(text), " - (wLAN)%.2e", this->bestWLand);
    sprintf(text + strlen(text), " [+ %g mm]", currentSize);
    reg_print_info(this->executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::PrintCurrentObjFunctionValue");
#endif
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
            dynamic_cast<F3dContent*>(this->con)->ZeroTransformationGradient();
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
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CorrectTransformation() {
    if (jacobianLogWeight > 0 && jacobianLogApproximation)
        ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::CorrectTransformation");
#endif
}
/* *************************************************************** */
template class reg_f3d<float>;
