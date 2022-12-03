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
 /* *************************************************************** */
template <class T>
reg_f3d<T>::reg_f3d(int refTimePoint, int floTimePoint)
    : reg_base<T>::reg_base(refTimePoint, floTimePoint) {

    executableName = (char *)"NiftyReg F3D";
    inputControlPointGrid = nullptr; // pointer to external
    controlPointGrid = nullptr;
    bendingEnergyWeight = 0.001;
    linearEnergyWeight = 0.00;
    jacobianLogWeight = 0.;
    jacobianLogApproximation = true;
    spacing[0] = -5;
    spacing[1] = std::numeric_limits<T>::quiet_NaN();
    spacing[2] = std::numeric_limits<T>::quiet_NaN();
    useConjGradient = true;
    useApproxGradient = false;

    // approxParzenWindow=true;

    // transformationGradient = nullptr;

    gridRefinement = true;

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::reg_f3d");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d<T>::~reg_f3d() {
    // DeallocateTransformationGradient();
    if (controlPointGrid != nullptr) {
        nifti_image_free(controlPointGrid);
        controlPointGrid = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::~reg_f3d");
#endif
}
/* *************************************************************** */
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
template <class T>
T reg_f3d<T>::InitialiseCurrentLevel(nifti_image *reference) {
    // Set the initial step size for the gradient ascent
    T maxStepSize = reference->dx > reference->dy ? reference->dx : reference->dy;
    if (reference->ndim > 2)
        maxStepSize = (reference->dz > maxStepSize) ? reference->dz : maxStepSize;

    // Refine the control point grid if required
    if (gridRefinement) {
        if (currentLevel == 0) {
            bendingEnergyWeight = bendingEnergyWeight / static_cast<T>(powf(16.0f, levelNumber - 1));
            linearEnergyWeight = linearEnergyWeight / static_cast<T>(powf(3.0f, levelNumber - 1));
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
// template <class T>
// void reg_f3d<T>::AllocateTransformationGradient() {
//     if (controlPointGrid == nullptr) {
//         reg_print_fct_error("reg_f3d<T>::AllocateTransformationGradient()");
//         reg_print_msg_error("The control point image is not defined");
//         reg_exit();
//     }
//     reg_f3d<T>::DeallocateTransformationGradient();
//     transformationGradient = nifti_copy_nim_info(controlPointGrid);
//     transformationGradient->data = (void*)calloc(transformationGradient->nvox,
//                                                        transformationGradient->nbyper);
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_f3d<T>::AllocateTransformationGradient");
// #endif
// }
/* *************************************************************** */
// template <class T>
// void reg_f3d<T>::DeallocateTransformationGradient() {
//     if (transformationGradient != nullptr) {
//         nifti_image_free(transformationGradient);
//         transformationGradient = nullptr;
//     }
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_f3d<T>::DeallocateTransformationGradient");
// #endif
// }
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CheckParameters() {
    reg_base<T>::CheckParameters();
    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    if (strcmp(executableName, "NiftyReg F3D") == 0 ||
        strcmp(executableName, "NiftyReg F3D GPU") == 0) {
        T penaltySum = bendingEnergyWeight +
            linearEnergyWeight +
            jacobianLogWeight +
            landmarkRegWeight;
        if (penaltySum >= 1.0) {
            similarityWeight = 0;
            similarityWeight /= penaltySum;
            bendingEnergyWeight /= penaltySum;
            linearEnergyWeight /= penaltySum;
            jacobianLogWeight /= penaltySum;
            landmarkRegWeight /= penaltySum;
        } else similarityWeight = 1.0 - penaltySum;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::CheckParameters");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::Initialise() {
    if (initialised) return;

    reg_base<T>::Initialise();

    // DETERMINE THE GRID SPACING AND CREATE THE GRID
    if (inputControlPointGrid == nullptr) {
        // Set the spacing along y and z if undefined. Their values are set to match
        // the spacing along the x axis
        if (spacing[1] != spacing[1]) spacing[1] = spacing[0];
        if (spacing[2] != spacing[2]) spacing[2] = spacing[0];

        /* Convert the spacing from voxel to mm if necessary */
        float spacingInMillimeter[3] = {spacing[0], spacing[1], spacing[2]};
        if (spacingInMillimeter[0] < 0) spacingInMillimeter[0] *= -1.0f * inputReference->dx;
        if (spacingInMillimeter[1] < 0) spacingInMillimeter[1] *= -1.0f * inputReference->dy;
        if (spacingInMillimeter[2] < 0) spacingInMillimeter[2] *= -1.0f * inputReference->dz;

        // Define the spacing for the first level
        float gridSpacing[3];
        gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(levelNumber - 1));
        gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(levelNumber - 1));
        gridSpacing[2] = 1.0f;
        if (referencePyramid[0]->nz > 1)
            gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(levelNumber - 1));

        // Create and allocate the control point image
        reg_createControlPointGrid<T>(&controlPointGrid, referencePyramid[0], gridSpacing);

        // The control point position image is initialised with the affine transformation
        if (affineTransformation == nullptr) {
            memset(controlPointGrid->data, 0, controlPointGrid->nvox * controlPointGrid->nbyper);
            reg_tools_multiplyValueToImage(controlPointGrid, controlPointGrid, 0.f);
            reg_getDeformationFromDisplacement(controlPointGrid);
        } else reg_affine_getDeformationField(affineTransformation, controlPointGrid);
    } else {
        // The control point grid image is initialised with the provided grid
        controlPointGrid = nifti_copy_nim_info(inputControlPointGrid);
        controlPointGrid->data = (void *)malloc(controlPointGrid->nvox * controlPointGrid->nbyper);
        memcpy(controlPointGrid->data, inputControlPointGrid->data,
               controlPointGrid->nvox * controlPointGrid->nbyper);
        // The final grid spacing is computed
        spacing[0] = controlPointGrid->dx / powf(2.0f, (float)(levelNumber - 1));
        spacing[1] = controlPointGrid->dy / powf(2.0f, (float)(levelNumber - 1));
        if (controlPointGrid->nz > 1)
            spacing[2] = controlPointGrid->dz / powf(2.0f, (float)(levelNumber - 1));
    }
#ifdef NDEBUG
    if (verbose) {
#endif
        std::string text;
        // Print out some global information about the registration
        reg_print_info(executableName, "***********************************************************");
        reg_print_info(executableName, "INPUT PARAMETERS");
        reg_print_info(executableName, "***********************************************************");
        reg_print_info(executableName, "Reference image:");
        text = stringFormat("\t* name: %s", inputReference->fname);
        reg_print_info(executableName, text.c_str());
        text = stringFormat("\t* image dimension: %i x %i x %i x %i",
                            inputReference->nx, inputReference->ny,
                            inputReference->nz, inputReference->nt);
        reg_print_info(executableName, text.c_str());
        text = stringFormat("\t* image spacing: %g x %g x %g mm",
                            inputReference->dx, inputReference->dy, inputReference->dz);
        reg_print_info(executableName, text.c_str());
        for (int i = 0; i < inputReference->nt; i++) {
            text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                                i, inputReference->nt - 1, referenceThresholdLow[i], referenceThresholdUp[i]);
            reg_print_info(executableName, text.c_str());
            if (measure_nmi != nullptr) {
                if (measure_nmi->GetTimepointsWeights()[i] > 0.0) {
                    text = stringFormat("\t* binnining size for timepoint %i/%i: %i",
                                        i, inputFloating->nt - 1, measure_nmi->GetReferenceBinNumber()[i] - 4);
                    reg_print_info(executableName, text.c_str());
                }
            }
        }
        text = stringFormat("\t* gaussian smoothing sigma: %g", referenceSmoothingSigma);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");
        reg_print_info(executableName, "Floating image:");
        reg_print_info(executableName, text.c_str());
        text = stringFormat("\t* name: %s", inputFloating->fname);
        reg_print_info(executableName, text.c_str());
        text = stringFormat("\t* image dimension: %i x %i x %i x %i",
                            inputFloating->nx, inputFloating->ny, inputFloating->nz, inputFloating->nt);
        reg_print_info(executableName, text.c_str());
        text = stringFormat("\t* image spacing: %g x %g x %g mm", inputFloating->dx,
                            inputFloating->dy, inputFloating->dz);
        reg_print_info(executableName, text.c_str());
        for (int i = 0; i < inputFloating->nt; i++) {
            text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                                i, inputFloating->nt - 1, floatingThresholdLow[i], floatingThresholdUp[i]);
            reg_print_info(executableName, text.c_str());
            if (measure_nmi != nullptr) {
                if (measure_nmi->GetTimepointsWeights()[i] > 0.0) {
                    text = stringFormat("\t* binning size for timepoint %i/%i: %i",
                                        i, inputFloating->nt - 1, measure_nmi->GetFloatingBinNumber()[i] - 4);
                    reg_print_info(executableName, text.c_str());
                }
            }
        }
        text = stringFormat("\t* gaussian smoothing sigma: %g", floatingSmoothingSigma);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");
        text = stringFormat("Warped image padding value: %g", warpedPaddingValue);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");
        text = stringFormat("Level number: %i", levelNumber);
        reg_print_info(executableName, text.c_str());
        if (levelNumber != levelToPerform) {
            text = stringFormat("\t* Level to perform: %i", levelToPerform);
            reg_print_info(executableName, text.c_str());
        }
        reg_print_info(executableName, "");
        text = stringFormat("Maximum iteration number during the last level: %i", (int)maxIterationNumber);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");

        text = stringFormat("Final spacing in mm: %g %g %g", spacing[0], spacing[1], spacing[2]);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");
        if (measure_ssd != nullptr)
            reg_print_info(executableName, "The SSD is used as a similarity measure.");
        if (measure_kld != nullptr)
            reg_print_info(executableName, "The KL divergence is used as a similarity measure.");
        if (measure_lncc != nullptr)
            reg_print_info(executableName, "The LNCC is used as a similarity measure.");
        if (measure_dti != nullptr)
            reg_print_info(executableName, "A DTI based measure is used as a similarity measure.");
        if (measure_mind != nullptr)
            reg_print_info(executableName, "MIND is used as a similarity measure.");
        if (measure_mindssc != nullptr)
            reg_print_info(executableName, "MINDSSC is used as a similarity measure.");
        if (measure_nmi != nullptr || (measure_dti == nullptr && measure_kld == nullptr &&
                                       measure_lncc == nullptr && measure_nmi == nullptr &&
                                       measure_ssd == nullptr && measure_mind == nullptr &&
                                       measure_mindssc == nullptr))
            reg_print_info(executableName, "The NMI is used as a similarity measure.");
        text = stringFormat("Similarity measure term weight: %g", similarityWeight);
        reg_print_info(executableName, text.c_str());
        reg_print_info(executableName, "");
        if (bendingEnergyWeight > 0) {
            text = stringFormat("Bending energy penalty term weight: %g", bendingEnergyWeight);
            reg_print_info(executableName, text.c_str());
            reg_print_info(executableName, "");
        }
        if ((linearEnergyWeight) > 0) {
            text = stringFormat("Linear energy penalty term weight: %g", linearEnergyWeight);
            reg_print_info(executableName, text.c_str());
            reg_print_info(executableName, "");
        }
        if (jacobianLogWeight > 0) {
            text = stringFormat("Jacobian-based penalty term weight: %g", jacobianLogWeight);
            reg_print_info(executableName, text.c_str());
            if (jacobianLogApproximation) {
                reg_print_info(executableName, "\t* Jacobian-based penalty term is approximated");
            } else {
                reg_print_info(executableName, "\t* Jacobian-based penalty term is not approximated");
            }
            reg_print_info(executableName, "");
        }
        if ((landmarkRegWeight) > 0) {
            text = stringFormat("Landmark distance regularisation term weight: %g", landmarkRegWeight);
            reg_print_info(executableName, text.c_str());
            reg_print_info(executableName, "");
        }
#ifdef NDEBUG
    }
#endif

    initialised = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::Initialise");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::InitContent(nifti_image *reference, nifti_image *floating, int *mask) {
    if (platformCode == NR_PLATFORM_CPU)
        con = new F3dContent(reference, floating, controlPointGrid, localWeightSimInput, mask, affineTransformation, sizeof(T));
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA)
        con = new CudaF3dContent(reference, floating, controlPointGrid, localWeightSimInput, mask, affineTransformation, sizeof(T));
#endif
    compute = platform->CreateCompute(con);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::DeinitContent() {
    delete compute;
    compute = nullptr;
    delete con;
    con = nullptr;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetDeformationField() {
    compute->GetDeformationField(false, // Composition
                                 true); // bspline
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetDeformationField");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type) {
    if (jacobianLogWeight <= 0) return 0;

    bool approx = type == 2 ? false : jacobianLogApproximation;

    double value = compute->GetJacobianPenaltyTerm(approx);

    unsigned int maxit = 5;
    if (type > 0) maxit = 20;
    unsigned int it = 0;
    while (value != value && it < maxit) {
        value = compute->CorrectFolding(approx);
#ifndef NDEBUG
        reg_print_msg_debug("Folding correction");
#endif
        it++;
    }
    if (type > 0) {
        if (value != value) {
            optimiser->RestoreBestDOF();
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
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm() {
    if (bendingEnergyWeight <= 0) return 0;

    double value = compute->ApproxBendingEnergy();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeBendingEnergyPenaltyTerm");
#endif
    return bendingEnergyWeight * value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm() {
    if (linearEnergyWeight <= 0)
        return 0;

    double value = compute->ApproxLinearEnergy();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeLinearEnergyPenaltyTerm");
#endif
    return linearEnergyWeight * value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm() {
    if (landmarkRegWeight <= 0)
        return 0;

    double value = compute->GetLandmarkDistance(landmarkRegNumber, landmarkReference, landmarkFloating);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm");
#endif
    return landmarkRegWeight * value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetSimilarityMeasureGradient() {
    GetVoxelBasedGradient();

    nifti_image *voxelBasedMeasureGradient = dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient();
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

    // Update the changes of voxelBasedMeasureGradient
    dynamic_cast<F3dContent*>(con)->SetVoxelBasedMeasureGradient(voxelBasedMeasureGradient);

    // The node based NMI gradient is extracted
    compute->VoxelCentricToNodeCentric(similarityWeight);

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetSimilarityMeasureGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetBendingEnergyGradient() {
    if (bendingEnergyWeight <= 0) return;

    compute->ApproxBendingEnergyGradient(bendingEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetBendingEnergyGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetLinearEnergyGradient() {
    if (linearEnergyWeight <= 0) return;

    compute->ApproxLinearEnergyGradient(linearEnergyWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetLinearEnergyGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetJacobianBasedGradient() {
    if (jacobianLogWeight <= 0) return;

    compute->JacobianPenaltyTermGradient(jacobianLogWeight, jacobianLogApproximation);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetLandmarkDistanceGradient() {
    if (landmarkRegWeight <= 0) return;

    compute->LandmarkDistanceGradient(landmarkRegNumber,
                                      landmarkReference,
                                      landmarkFloating,
                                      landmarkRegWeight);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetLandmarkDistanceGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
// template <class T>
// void reg_f3d<T>::SetGradientImageToZero() {
//     T* nodeGradPtr = static_cast<T*>(transformationGradient->data);
//     for (size_t i = 0; i < transformationGradient->nvox; ++i)
//         *nodeGradPtr++ = 0;
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_f3d<T>::SetGradientImageToZero");
// #endif
// }
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_f3d<T>::NormaliseGradient() {
    // First compute the gradient max length for normalisation purpose
    T maxGradLength = (T)compute->GetMaximalLength(optimiseX, optimiseY, optimiseZ);

    if (strcmp(executableName, "NiftyReg F3D") == 0) {
        // The gradient is normalised if we are running f3d
        // It will be normalised later when running f3d_sym or f3d2
        compute->NormaliseGradient(maxGradLength);
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
/* *************************************************************** */
template <class T>
void reg_f3d<T>::DisplayCurrentLevelParameters() {
#ifdef NDEBUG
    if (verbose) {
#endif
        nifti_image *reference = con->Content::GetReference();
        nifti_image *floating = con->Content::GetFloating();
        char text[255];
        sprintf(text, "Current level: %i / %i", currentLevel + 1, levelNumber);
        reg_print_info(executableName, text);
        sprintf(text, "Maximum iteration number: %i", (int)maxIterationNumber);
        reg_print_info(executableName, text);
        reg_print_info(executableName, "Current reference image");
        sprintf(text, "\t* image dimension: %i x %i x %i x %i", reference->nx, reference->ny, reference->nz, reference->nt);
        reg_print_info(executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm", reference->dx, reference->dy, reference->dz);
        reg_print_info(executableName, text);
        reg_print_info(executableName, "Current floating image");
        sprintf(text, "\t* image dimension: %i x %i x %i x %i", floating->nx, floating->ny, floating->nz, floating->nt);
        reg_print_info(executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm", floating->dx, floating->dy, floating->dz);
        reg_print_info(executableName, text);
        reg_print_info(executableName, "Current control point image");
        sprintf(text, "\t* image dimension: %i x %i x %i",
                controlPointGrid->nx, controlPointGrid->ny,
                controlPointGrid->nz);
        reg_print_info(executableName, text);
        sprintf(text, "\t* image spacing: %g x %g x %g mm",
                controlPointGrid->dx, controlPointGrid->dy,
                controlPointGrid->dz);
        reg_print_info(executableName, text);
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
/* *************************************************************** */
template <class T>
double reg_f3d<T>::GetObjectiveFunctionValue() {
    currentWJac = ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

    currentWBE = ComputeBendingEnergyPenaltyTerm();

    currentWLE = ComputeLinearEnergyPenaltyTerm();

    currentWLand = ComputeLandmarkDistancePenaltyTerm();

    // Compute initial similarity measure
    currentWMeasure = 0.0;
    if (similarityWeight > 0) {
        WarpFloatingImage(interpolation);
        currentWMeasure = ComputeSimilarityMeasure();
    }
#ifndef NDEBUG
    char text[255];
    sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g | (wLan) %g",
            currentWMeasure, currentWBE, currentWLE, currentWJac, currentWLand);
    reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif
    // Store the global objective function value

    return currentWMeasure - currentWBE - currentWLE - currentWJac - currentWLand;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::UpdateParameters(float scale) {
    T *currentDOF = optimiser->GetCurrentDOF();
    T *bestDOF = optimiser->GetBestDOF();
    T *gradient = optimiser->GetGradient();

    compute->UpdateControlPointPosition(currentDOF, bestDOF, gradient, scale, optimiseX, optimiseY, optimiseZ);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::UpdateParameters");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::SetOptimiser() {
    optimiser = platform->CreateOptimiser<T>(dynamic_cast<F3dContent*>(con),
                                             this,
                                             maxIterationNumber,
                                             useConjGradient,
                                             optimiseX,
                                             optimiseY,
                                             optimiseZ);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::SmoothGradient() {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    // The gradient is smoothed using a Gaussian kernel if it is required
    if (gradientSmoothingSigma != 0) {
        float kernel = fabs(gradientSmoothingSigma);
        F3dContent *con = dynamic_cast<F3dContent*>(this->con);
        reg_tools_kernelConvolution(con->GetTransformationGradient(), &kernel, GAUSSIAN_KERNEL);
        // Update the changes of transformationGradient
        con->SetTransformationGradient(con->F3dContent::GetTransformationGradient());
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
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
        T currentValue = optimiser->GetBestDOF()[i];
        gridPtr[i] = currentValue + eps;
        // Update the changes. Bad hack, fix that!
        con->SetControlPointGrid(controlPointGrid);
        double valPlus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue - eps;
        // Update the changes. Bad hack, fix that!
        con->SetControlPointGrid(controlPointGrid);
        double valMinus = GetObjectiveFunctionValue();
        gridPtr[i] = currentValue;
        // Update the changes. Bad hack, fix that!
        con->SetControlPointGrid(controlPointGrid);
        gradPtr[i] = -(T)((valPlus - valMinus) / (2.0 * eps));
    }

    // Update the changes
    con->SetTransformationGradient(transformationGradient);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
nifti_image** reg_f3d<T>::GetWarpedImage() {
    // The initial images are used
    if (!inputReference || !inputFloating || !controlPointGrid) {
        reg_print_fct_error("reg_f3d<T>::GetWarpedImage()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    const int datatype = inputFloating->datatype;

    InitContent(inputReference, inputFloating, nullptr);

    WarpFloatingImage(3); // cubic spline interpolation

    nifti_image **warpedImage = (nifti_image**)calloc(2, sizeof(nifti_image*));
    warpedImage[0] = con->GetWarped(datatype, 0);
    if (inputFloating->nt == 2)
        warpedImage[1] = con->GetWarped(datatype, 1);

    con->SetWarped(nullptr); // Prevent deallocating of warpedImage
    DeinitContent();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetWarpedImage");
#endif
    return warpedImage;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
nifti_image* reg_f3d<T>::GetControlPointPositionImage() {
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(controlPointGrid);
    returnedControlPointGrid->data = (void*)malloc(returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    memcpy(returnedControlPointGrid->data, controlPointGrid->data,
           returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
    return returnedControlPointGrid;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetControlPointPositionImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateBestObjFunctionValue() {
    bestWMeasure = currentWMeasure;
    bestWBE = currentWBE;
    bestWLE = currentWLE;
    bestWJac = currentWJac;
    bestWLand = currentWLand;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintInitialObjFunctionValue() {
    if (!verbose) return;

    double bestValue = optimiser->GetBestObjFunctionValue();

    char text[255];
    sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wLAN)%g",
            bestValue, bestWMeasure, bestWBE, bestWLE, bestWJac, bestWLand);
    reg_print_info(executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::PrintInitialObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintCurrentObjFunctionValue(T currentSize) {
    if (!verbose) return;

    char text[255];
    sprintf(text, "[%i] Current objective function: %g",
            (int)optimiser->GetCurrentIterationNumber(),
            optimiser->GetBestObjFunctionValue());
    sprintf(text + strlen(text), " = (wSIM)%g", bestWMeasure);
    if (bendingEnergyWeight > 0)
        sprintf(text + strlen(text), " - (wBE)%.2e", bestWBE);
    if (linearEnergyWeight > 0)
        sprintf(text + strlen(text), " - (wLE)%.2e", bestWLE);
    if (jacobianLogWeight > 0)
        sprintf(text + strlen(text), " - (wJAC)%.2e", bestWJac);
    if (landmarkRegWeight > 0)
        sprintf(text + strlen(text), " - (wLAN)%.2e", bestWLand);
    sprintf(text + strlen(text), " [+ %g mm]", currentSize);
    reg_print_info(executableName, text);
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::PrintCurrentObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetObjectiveFunctionGradient() {
    if (!useApproxGradient) {
        // Compute the gradient of the similarity measure
        if (similarityWeight > 0) {
            WarpFloatingImage(interpolation);
            GetSimilarityMeasureGradient();
        } else {
            dynamic_cast<F3dContent*>(con)->ZeroTransformationGradient();
        }
        // Compute the penalty term gradients if required
        GetBendingEnergyGradient();
        GetJacobianBasedGradient();
        GetLinearEnergyGradient();
        GetLandmarkDistanceGradient();
    } else {
        GetApproximatedGradient();
    }

    optimiser->IncrementCurrentIterationNumber();

    // Smooth the gradient if require
    SmoothGradient();
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionGradient");
#endif
}
/* *************************************************************** */
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
/* *************************************************************** */

template class reg_f3d<float>;
