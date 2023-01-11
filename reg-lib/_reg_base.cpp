/**
 * @file _reg_base.cpp
 * @author Marc Modat
 * @date 15/11/2012
 *
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_base.h"
#include "F3dContent.h" // TODO Temporary fix! Remove this line!

/* *************************************************************** */
/* *************************************************************** */
template<class T>
reg_base<T>::reg_base(int refTimePoint, int floTimePoint) {
    platform = nullptr;
    platformCode = NR_PLATFORM_CPU;
    gpuIdx = 999;

    optimiser = nullptr;
    maxIterationNumber = 150;
    optimiseX = true;
    optimiseY = true;
    optimiseZ = true;
    perturbationNumber = 0;
    useConjGradient = true;
    useApproxGradient = false;

    measure_ssd = nullptr;
    measure_kld = nullptr;
    measure_dti = nullptr;
    measure_lncc = nullptr;
    measure_nmi = nullptr;
    measure_mind = nullptr;
    measure_mindssc = nullptr;
    localWeightSimInput = nullptr;
    // localWeightSimCurrent = nullptr;

    similarityWeight = 0; // automatically set depending of the penalty term weights

    executableName = (char*)"NiftyReg BASE";
    referenceTimePoint = refTimePoint;
    floatingTimePoint = floTimePoint;
    inputReference = nullptr; // pointer to external
    inputFloating = nullptr; // pointer to external
    maskImage = nullptr; // pointer to external
    affineTransformation = nullptr;  // pointer to external
    referenceMask = nullptr;
    referenceSmoothingSigma = 0;
    floatingSmoothingSigma = 0;
    referenceThresholdUp = new float[referenceTimePoint];
    referenceThresholdLow = new float[referenceTimePoint];
    floatingThresholdUp = new float[floatingTimePoint];
    floatingThresholdLow = new float[floatingTimePoint];
    for (int i = 0; i < referenceTimePoint; i++) {
        referenceThresholdUp[i] = std::numeric_limits<T>::max();
        referenceThresholdLow[i] = -std::numeric_limits<T>::max();
    }
    for (int i = 0; i < floatingTimePoint; i++) {
        floatingThresholdUp[i] = std::numeric_limits<T>::max();
        floatingThresholdLow[i] = -std::numeric_limits<T>::max();
    }
    robustRange = false;
    warpedPaddingValue = std::numeric_limits<T>::quiet_NaN();
    levelNumber = 3;
    levelToPerform = 0;
    gradientSmoothingSigma = 0;
    verbose = true;
    usePyramid = true;
    forwardJacobianMatrix = nullptr;

    initialised = false;
    referencePyramid = nullptr;
    floatingPyramid = nullptr;
    maskPyramid = nullptr;
    activeVoxelNumber = nullptr;
    // reference = nullptr;
    // floating = nullptr;
    // currentMask = nullptr;
    // warped = nullptr;
    // deformationFieldImage = nullptr;
    // warpedGradient = nullptr;
    // voxelBasedMeasureGradient = nullptr;

    interpolation = 1;

    landmarkRegWeight = 0;
    landmarkRegNumber = 0;
    landmarkReference = nullptr;
    landmarkFloating = nullptr;

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::reg_base");
#endif
}
/* *************************************************************** */
template<class T>
reg_base<T>::~reg_base() {
    // DeallocateWarped();
    // DeallocateWarpedGradient();
    // DeallocateDeformationField();
    // DeallocateVoxelBasedMeasureGradient();
    if (referencePyramid != nullptr) {
        if (usePyramid) {
            for (unsigned int i = 0; i < levelToPerform; i++) {
                if (referencePyramid[i] != nullptr) {
                    nifti_image_free(referencePyramid[i]);
                    referencePyramid[i] = nullptr;
                }
            }
        } else {
            if (referencePyramid[0] != nullptr) {
                nifti_image_free(referencePyramid[0]);
                referencePyramid[0] = nullptr;
            }
        }
        free(referencePyramid);
        referencePyramid = nullptr;
    }
    if (maskPyramid != nullptr) {
        if (usePyramid) {
            for (unsigned int i = 0; i < levelToPerform; i++) {
                if (maskPyramid[i] != nullptr) {
                    free(maskPyramid[i]);
                    maskPyramid[i] = nullptr;
                }
            }
        } else {
            if (maskPyramid[0] != nullptr) {
                free(maskPyramid[0]);
                maskPyramid[0] = nullptr;
            }
        }
        free(maskPyramid);
        maskPyramid = nullptr;
    }
    if (floatingPyramid != nullptr) {
        if (usePyramid) {
            for (unsigned int i = 0; i < levelToPerform; i++) {
                if (floatingPyramid[i] != nullptr) {
                    nifti_image_free(floatingPyramid[i]);
                    floatingPyramid[i] = nullptr;
                }
            }
        } else {
            if (floatingPyramid[0] != nullptr) {
                nifti_image_free(floatingPyramid[0]);
                floatingPyramid[0] = nullptr;
            }
        }
        free(floatingPyramid);
        floatingPyramid = nullptr;
    }
    if (activeVoxelNumber != nullptr) {
        free(activeVoxelNumber);
        activeVoxelNumber = nullptr;
    }
    if (referenceThresholdUp != nullptr) {
        delete[]referenceThresholdUp;
        referenceThresholdUp = nullptr;
    }
    if (referenceThresholdLow != nullptr) {
        delete[]referenceThresholdLow;
        referenceThresholdLow = nullptr;
    }
    if (floatingThresholdUp != nullptr) {
        delete[]floatingThresholdUp;
        floatingThresholdUp = nullptr;
    }
    if (floatingThresholdLow != nullptr) {
        delete[]floatingThresholdLow;
        floatingThresholdLow = nullptr;
    }
    if (optimiser != nullptr) {
        delete optimiser;
        optimiser = nullptr;
    }

    if (measure_nmi != nullptr)
        delete measure_nmi;
    if (measure_ssd != nullptr)
        delete measure_ssd;
    if (measure_kld != nullptr)
        delete measure_kld;
    if (measure_dti != nullptr)
        delete measure_dti;
    if (measure_lncc != nullptr)
        delete measure_lncc;
    if (measure_mind != nullptr)
        delete measure_mind;
    if (measure_mindssc != nullptr)
        delete measure_mindssc;

    delete platform;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::~reg_base");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceImage(nifti_image *r) {
    inputReference = r;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingImage(nifti_image *f) {
    inputFloating = f;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned int iter) {
    maxIterationNumber = iter;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetMaximalIterationNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceMask(nifti_image *m) {
    maskImage = m;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceMask");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetAffineTransformation(mat44 *a) {
    affineTransformation = a;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetAffineTransformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T s) {
    referenceSmoothingSigma = s;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T s) {
    floatingSmoothingSigma = s;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned int i, T t) {
    referenceThresholdUp[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned int i, T t) {
    referenceThresholdLow[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceThresholdLow");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned int i, T t) {
    floatingThresholdUp[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned int i, T t) {
    floatingThresholdLow[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingThresholdLow");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseRobustRange() {
    robustRange = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseRobustRange() {
    robustRange = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(float p) {
    warpedPaddingValue = p;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetWarpedPaddingValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned int l) {
    levelNumber = l;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLevelNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned int l) {
    levelToPerform = l;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLevelToPerform");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T g) {
    gradientSmoothingSigma = g;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetGradientSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseConjugateGradient() {
    useConjGradient = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient() {
    useConjGradient = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseApproximatedGradient() {
    useApproxGradient = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient() {
    useApproxGradient = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::PrintOutInformation() {
    verbose = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::PrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotPrintOutInformation() {
    verbose = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotPrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach() {
    usePyramid = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUsePyramidalApproach");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNearestNeighborInterpolation() {
    interpolation = 0;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNearestNeighborInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLinearInterpolation() {
    interpolation = 1;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseLinearInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation() {
    interpolation = 3;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseCubicSplineInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLandmarkRegularisationParam(size_t n, float *r, float *f, float w) {
    landmarkRegNumber = n;
    landmarkReference = r;
    landmarkFloating = f;
    landmarkRegWeight = w;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLandmarkRegularisationParam");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
// template<class T>
// void reg_base<T>::DeallocateCurrentInputImage() {
//     reference = nullptr;
//     currentMask = nullptr;
//     floating = nullptr;
//     if (localWeightSimCurrent != nullptr)
//         nifti_image_free(localWeightSimCurrent);
//     localWeightSimCurrent = nullptr;
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::DeallocateCurrentInputImage");
// #endif
// }
/* *************************************************************** */
/* *************************************************************** */
// template<class T>
// void reg_base<T>::AllocateWarped() {
//     if (reference == nullptr) {
//         reg_print_fct_error("reg_base::AllocateWarped()");
//         reg_print_msg_error("The reference image is not defined");
//         reg_exit();
//     }
//     reg_base<T>::DeallocateWarped();
//     warped = nifti_copy_nim_info(reference);
//     warped->dim[0] = warped->ndim = floating->ndim;
//     warped->dim[4] = warped->nt = floating->nt;
//     warped->pixdim[4] = warped->dt = 1;
//     warped->nvox = (size_t)(warped->nx * warped->ny * warped->nz * warped->nt);
//     warped->scl_slope = 1;
//     warped->scl_inter = 0;
//     warped->datatype = floating->datatype;
//     warped->nbyper = floating->nbyper;
//     warped->data = (void*)calloc(warped->nvox, warped->nbyper);
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::AllocateWarped");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::DeallocateWarped() {
//     if (warped != nullptr)
//         nifti_image_free(warped);
//     warped = nullptr;
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::DeallocateWarped");
// #endif
// }
/* *************************************************************** */
/* *************************************************************** */
// template<class T>
// void reg_base<T>::AllocateDeformationField() {
//     if (reference == nullptr) {
//         reg_print_fct_error("reg_base::AllocateDeformationField()");
//         reg_print_msg_error("The reference image is not defined");
//         reg_exit();
//     }
//     reg_base<T>::DeallocateDeformationField();
//     deformationFieldImage = nifti_copy_nim_info(reference);
//     deformationFieldImage->dim[0] = deformationFieldImage->ndim = 5;
//     // deformationFieldImage->dim[1] = deformationFieldImage->nx;
//     // deformationFieldImage->dim[2] = deformationFieldImage->ny;
//     // deformationFieldImage->dim[3] = deformationFieldImage->nz;
//     deformationFieldImage->dim[4] = deformationFieldImage->nt = 1;
//     deformationFieldImage->pixdim[4] = deformationFieldImage->dt = 1.0;
//     if (reference->nz == 1)
//         deformationFieldImage->dim[5] = deformationFieldImage->nu = 2;
//     else deformationFieldImage->dim[5] = deformationFieldImage->nu = 3;
//     deformationFieldImage->pixdim[5] = deformationFieldImage->du = 1.0;
//     deformationFieldImage->dim[6] = deformationFieldImage->nv = 1;
//     deformationFieldImage->pixdim[6] = deformationFieldImage->dv = 1.0;
//     deformationFieldImage->dim[7] = deformationFieldImage->nw = 1;
//     deformationFieldImage->pixdim[7] = deformationFieldImage->dw = 1.0;
//     deformationFieldImage->nvox =
//         (size_t)deformationFieldImage->nx *
//         (size_t)deformationFieldImage->ny *
//         (size_t)deformationFieldImage->nz *
//         (size_t)deformationFieldImage->nt *
//         (size_t)deformationFieldImage->nu;
//     deformationFieldImage->nbyper = sizeof(T);
//     if (sizeof(T) == sizeof(float))
//         deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
//     else deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
//     deformationFieldImage->data = (void*)calloc(deformationFieldImage->nvox,
//                                                 deformationFieldImage->nbyper);
//     deformationFieldImage->intent_code = NIFTI_INTENT_VECTOR;
//     memset(deformationFieldImage->intent_name, 0, 16);
//     strcpy(deformationFieldImage->intent_name, "NREG_TRANS");
//     deformationFieldImage->intent_p1 = DEF_FIELD;
//     deformationFieldImage->scl_slope = 1;
//     deformationFieldImage->scl_inter = 0;

//     if (measure_dti != nullptr)
//         forwardJacobianMatrix = (mat33*)malloc(deformationFieldImage->nx * deformationFieldImage->ny *
//                                                deformationFieldImage->nz * sizeof(mat33));
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::AllocateDeformationField");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::DeallocateDeformationField() {
//     if (deformationFieldImage != nullptr) {
//         nifti_image_free(deformationFieldImage);
//         deformationFieldImage = nullptr;
//     }
//     if (forwardJacobianMatrix != nullptr)
//         free(forwardJacobianMatrix);
//     forwardJacobianMatrix = nullptr;
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::DeallocateDeformationField");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::AllocateWarpedGradient() {
//     if (deformationFieldImage == nullptr) {
//         reg_print_fct_error("reg_base::AllocateWarpedGradient()");
//         reg_print_msg_error("The deformation field image is not defined");
//         reg_exit();
//     }
//     reg_base<T>::DeallocateWarpedGradient();
//     warpedGradient = nifti_copy_nim_info(deformationFieldImage);
//     warpedGradient->data = (void*)calloc(warpedGradient->nvox,
//                                          warpedGradient->nbyper);
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::AllocateWarpedGradient");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::DeallocateWarpedGradient() {
//     if (warpedGradient != nullptr) {
//         nifti_image_free(warpedGradient);
//         warpedGradient = nullptr;
//     }
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::DeallocateWarpedGradient");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::AllocateVoxelBasedMeasureGradient() {
//     if (deformationFieldImage == nullptr) {
//         reg_print_fct_error("reg_base::AllocateVoxelBasedMeasureGradient()");
//         reg_print_msg_error("The deformation field image is not defined");
//         reg_exit();
//     }
//     reg_base<T>::DeallocateVoxelBasedMeasureGradient();
//     voxelBasedMeasureGradient = nifti_copy_nim_info(deformationFieldImage);
//     voxelBasedMeasureGradient->data = (void*)calloc(voxelBasedMeasureGradient->nvox,
//                                                     voxelBasedMeasureGradient->nbyper);
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::AllocateVoxelBasedMeasureGradient");
// #endif
// }
/* *************************************************************** */
// template<class T>
// void reg_base<T>::DeallocateVoxelBasedMeasureGradient() {
//     if (voxelBasedMeasureGradient != nullptr) {
//         nifti_image_free(voxelBasedMeasureGradient);
//         voxelBasedMeasureGradient = nullptr;
//     }
// #ifndef NDEBUG
//     reg_print_fct_debug("reg_base<T>::DeallocateVoxelBasedMeasureGradient");
// #endif
// }
/* *************************************************************** */
template<class T>
void reg_base<T>::CheckParameters() {
    // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
    if (inputReference == nullptr) {
        reg_print_fct_error("reg_base::CheckParameters()");
        reg_print_msg_error("The reference image is not defined");
        reg_exit();
    }
    if (inputFloating == nullptr) {
        reg_print_fct_error("reg_base::CheckParameters()");
        reg_print_msg_error("The floating image is not defined");
        reg_exit();
    }

    // CHECK THE MASK DIMENSION IF IT IS DEFINED
    if (maskImage != nullptr) {
        if (inputReference->nx != maskImage->nx ||
            inputReference->ny != maskImage->ny ||
            inputReference->nz != maskImage->nz) {
            reg_print_fct_error("reg_base::CheckParameters()");
            reg_print_msg_error("The reference and mask images have different dimension");
            reg_exit();
        }
    }

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if (levelToPerform > 0) {
        levelToPerform = levelToPerform < levelNumber ? levelToPerform : levelNumber;
    } else levelToPerform = levelNumber;
    if (levelToPerform == 0 || levelToPerform > levelNumber)
        levelToPerform = levelNumber;

    // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
    if (measure_nmi == nullptr &&
        measure_ssd == nullptr &&
        measure_dti == nullptr &&
        measure_lncc == nullptr &&
        measure_lncc == nullptr &&
        measure_kld == nullptr &&
        measure_mind == nullptr &&
        measure_mindssc == nullptr) {
        measure_nmi = new reg_nmi;
        for (int i = 0; i < inputReference->nt; ++i)
            measure_nmi->SetTimepointWeight(i, 1.0);
    }

    // CHECK THAT IMAGES HAVE SAME NUMBER OF CHANNELS (TIMEPOINTS)
    // THAT EACH CHANNEL HAS AT LEAST ONE SIMILARITY MEASURE ASSIGNED
    // AND THAT EACH SIMILARITY MEASURE IS USED FOR AT LEAST ONE CHANNEL
    // NORMALISE CHANNEL AND SIMILARITY WEIGHTS SO TOTAL = 1
    //
    // NOTE - DTI currently ignored as needs fixing
    //
    // tests ignored if using MIND or MINDSSD as they are not implemented for multi-channel or weighting
    if (measure_mind == nullptr && measure_mindssc == nullptr) {
        if (inputFloating->nt != inputReference->nt) {
            reg_print_fct_error("reg_base::CheckParameters()");
            reg_print_msg_error("The reference and floating images have different numbers of channels (timepoints)");
            reg_exit();
        }
        double *chanWeightSum = new double[inputReference->nt]();
        double simWeightSum, totWeightSum = 0.;
        double *nmiWeights = nullptr, *ssdWeights = nullptr, *kldWeights = nullptr, *lnccWeights = nullptr;
        if (measure_nmi != nullptr) {
            nmiWeights = measure_nmi->GetTimepointsWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (nmiWeights[n] < 0) {
                    char text[255];
                    sprintf(text, "The NMI weight for timepoint %d has a negative value - weights must be positive", n);
                    reg_print_fct_error("reg_base::CheckParameters()");
                    reg_print_msg_error(text);
                    reg_exit();
                }
                chanWeightSum[n] += nmiWeights[n];
                simWeightSum += nmiWeights[n];
                totWeightSum += nmiWeights[n];
            }
            if (simWeightSum == 0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The NMI similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (measure_ssd != nullptr) {
            ssdWeights = measure_ssd->GetTimepointsWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (ssdWeights[n] < 0) {
                    char text[255];
                    sprintf(text, "The SSD weight for timepoint %d has a negative value - weights must be positive", n);
                    reg_print_fct_error("reg_base::CheckParameters()");
                    reg_print_msg_error(text);
                    reg_exit();
                }
                chanWeightSum[n] += ssdWeights[n];
                simWeightSum += ssdWeights[n];
                totWeightSum += ssdWeights[n];
            }
            if (simWeightSum == 0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The SSD similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (measure_kld != nullptr) {
            kldWeights = measure_kld->GetTimepointsWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (kldWeights[n] < 0) {
                    char text[255];
                    sprintf(text, "The KLD weight for timepoint %d has a negative value - weights must be positive", n);
                    reg_print_fct_error("reg_base::CheckParameters()");
                    reg_print_msg_error(text);
                    reg_exit();
                }
                chanWeightSum[n] += kldWeights[n];
                simWeightSum += kldWeights[n];
                totWeightSum += kldWeights[n];
            }
            if (simWeightSum == 0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The KLD similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (measure_lncc != nullptr) {
            lnccWeights = measure_lncc->GetTimepointsWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (lnccWeights[n] < 0) {
                    char text[255];
                    sprintf(text, "The LNCC weight for timepoint %d has a negative value - weights must be positive", n);
                    reg_print_fct_error("reg_base::CheckParameters()");
                    reg_print_msg_error(text);
                    reg_exit();
                }
                chanWeightSum[n] += lnccWeights[n];
                simWeightSum += lnccWeights[n];
                totWeightSum += lnccWeights[n];
            }
            if (simWeightSum == 0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The LNCC similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        for (int n = 0; n < inputReference->nt; n++) {
            if (chanWeightSum[n] == 0) {
                char text[255];
                sprintf(text, "Channel %d has a weight of 0 for all similarity measures so will be ignored", n);
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn(text);
            }
            if (measure_nmi != nullptr)
                measure_nmi->SetTimepointWeight(n, nmiWeights[n] / totWeightSum);
            if (measure_ssd != nullptr)
                measure_ssd->SetTimepointWeight(n, ssdWeights[n] / totWeightSum);
            if (measure_kld != nullptr)
                measure_kld->SetTimepointWeight(n, kldWeights[n] / totWeightSum);
            if (measure_lncc != nullptr)
                measure_lncc->SetTimepointWeight(n, lnccWeights[n] / totWeightSum);
        }
        delete[] chanWeightSum;
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::CheckParameters");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::InitialiseSimilarity() {
    // TODO Update this section to handle CUDA
    // TODO Move this function to reg_f3d
    if (measure_nmi != nullptr)
        measure_nmi->InitialiseMeasure(con->GetReference(),
                                       con->GetFloating(),
                                       con->GetReferenceMask(),
                                       con->GetWarped(),
                                       dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_ssd != nullptr)
        measure_ssd->InitialiseMeasure(con->GetReference(),
                                       con->GetFloating(),
                                       con->GetReferenceMask(),
                                       con->GetWarped(),
                                       dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_kld != nullptr)
        measure_kld->InitialiseMeasure(con->GetReference(),
                                       con->GetFloating(),
                                       con->GetReferenceMask(),
                                       con->GetWarped(),
                                       dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_lncc != nullptr)
        measure_lncc->InitialiseMeasure(con->GetReference(),
                                        con->GetFloating(),
                                        con->GetReferenceMask(),
                                        con->GetWarped(),
                                        dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                        dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                        dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_dti != nullptr)
        measure_dti->InitialiseMeasure(con->GetReference(),
                                       con->GetFloating(),
                                       con->GetReferenceMask(),
                                       con->GetWarped(),
                                       dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                       dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_mind != nullptr)
        measure_mind->InitialiseMeasure(con->GetReference(),
                                        con->GetFloating(),
                                        con->GetReferenceMask(),
                                        con->GetWarped(),
                                        dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                        dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                        dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

    if (measure_mindssc != nullptr)
        measure_mindssc->InitialiseMeasure(con->GetReference(),
                                           con->GetFloating(),
                                           con->GetReferenceMask(),
                                           con->GetWarped(),
                                           dynamic_cast<F3dContent*>(con)->GetWarpedGradient(),
                                           dynamic_cast<F3dContent*>(con)->GetVoxelBasedMeasureGradient(),
                                           dynamic_cast<F3dContent*>(con)->GetLocalWeightSim());

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::InitialiseSimilarity");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::Initialise() {
    if (initialised) return;

    CheckParameters();

    platform = new Platform(platformCode);
    platform->SetGpuIdx(gpuIdx);

    // CREATE THE PYRAMID IMAGES
    if (usePyramid) {
        referencePyramid = (nifti_image**)malloc(levelToPerform * sizeof(nifti_image*));
        floatingPyramid = (nifti_image**)malloc(levelToPerform * sizeof(nifti_image*));
        maskPyramid = (int**)malloc(levelToPerform * sizeof(int*));
        activeVoxelNumber = (int*)malloc(levelToPerform * sizeof(int));
    } else {
        referencePyramid = (nifti_image**)malloc(sizeof(nifti_image*));
        floatingPyramid = (nifti_image**)malloc(sizeof(nifti_image*));
        maskPyramid = (int**)malloc(sizeof(int*));
        activeVoxelNumber = (int*)malloc(sizeof(int));
    }

    // Update the input images threshold if required
    if (robustRange) {
        // Create a copy of the reference image to extract the robust range
        nifti_image *temp_reference = nifti_copy_nim_info(inputReference);
        temp_reference->data = (void*)malloc(temp_reference->nvox * temp_reference->nbyper);
        memcpy(temp_reference->data, inputReference->data, temp_reference->nvox * temp_reference->nbyper);
        reg_tools_changeDatatype<T>(temp_reference);
        // Extract the robust range of the reference image
        T *refDataPtr = static_cast<T *>(temp_reference->data);
        reg_heapSort(refDataPtr, temp_reference->nvox);
        // Update the reference threshold values if no value has been setup by the user
        if (referenceThresholdLow[0] == -std::numeric_limits<T>::max())
            referenceThresholdLow[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox * 0.02f)];
        if (referenceThresholdUp[0] == std::numeric_limits<T>::max())
            referenceThresholdUp[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox * 0.98f)];
        // Free the temporarily allocated image
        nifti_image_free(temp_reference);

        // Create a copy of the floating image to extract the robust range
        nifti_image *temp_floating = nifti_copy_nim_info(inputFloating);
        temp_floating->data = (void*)malloc(temp_floating->nvox * temp_floating->nbyper);
        memcpy(temp_floating->data, inputFloating->data, temp_floating->nvox * temp_floating->nbyper);
        reg_tools_changeDatatype<T>(temp_floating);
        // Extract the robust range of the floating image
        T *floDataPtr = static_cast<T *>(temp_floating->data);
        reg_heapSort(floDataPtr, temp_floating->nvox);
        // Update the floating threshold values if no value has been setup by the user
        if (floatingThresholdLow[0] == -std::numeric_limits<T>::max())
            floatingThresholdLow[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox * 0.02f)];
        if (floatingThresholdUp[0] == std::numeric_limits<T>::max())
            floatingThresholdUp[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox * 0.98f)];
        // Free the temporarily allocated image
        nifti_image_free(temp_floating);
    }

    // FINEST LEVEL OF REGISTRATION
    if (usePyramid) {
        reg_createImagePyramid<T>(inputReference, referencePyramid, levelNumber, levelToPerform);
        reg_createImagePyramid<T>(inputFloating, floatingPyramid, levelNumber, levelToPerform);
        if (maskImage != nullptr)
            reg_createMaskPyramid<T>(maskImage, maskPyramid, levelNumber, levelToPerform, activeVoxelNumber);
        else {
            for (unsigned int l = 0; l < levelToPerform; ++l) {
                activeVoxelNumber[l] = referencePyramid[l]->nx * referencePyramid[l]->ny * referencePyramid[l]->nz;
                maskPyramid[l] = (int*)calloc(activeVoxelNumber[l], sizeof(int));
            }
        }
    } else {
        reg_createImagePyramid<T>(inputReference, referencePyramid, 1, 1);
        reg_createImagePyramid<T>(inputFloating, floatingPyramid, 1, 1);
        if (maskImage != nullptr)
            reg_createMaskPyramid<T>(maskImage, maskPyramid, 1, 1, activeVoxelNumber);
        else {
            activeVoxelNumber[0] = referencePyramid[0]->nx * referencePyramid[0]->ny * referencePyramid[0]->nz;
            maskPyramid[0] = (int*)calloc(activeVoxelNumber[0], sizeof(int));
        }
    }

    unsigned int pyramidalLevelNumber = 1;
    if (usePyramid) pyramidalLevelNumber = levelToPerform;

    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < levelToPerform; l++) {
        if (referenceSmoothingSigma != 0) {
            bool *active = new bool[referencePyramid[l]->nt];
            float *sigma = new float[referencePyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < referencePyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = referenceSmoothingSigma;
            reg_tools_kernelConvolution(referencePyramid[l], sigma, GAUSSIAN_KERNEL, nullptr, active);
            delete[]active;
            delete[]sigma;
        }
        if (floatingSmoothingSigma != 0) {
            // Only the first image is smoothed
            bool *active = new bool[floatingPyramid[l]->nt];
            float *sigma = new float[floatingPyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < floatingPyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = floatingSmoothingSigma;
            reg_tools_kernelConvolution(floatingPyramid[l], sigma, GAUSSIAN_KERNEL, nullptr, active);
            delete[]active;
            delete[]sigma;
        }
    }

    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < pyramidalLevelNumber; l++) {
        reg_thresholdImage<T>(referencePyramid[l], referenceThresholdLow[0], referenceThresholdUp[0]);
        reg_thresholdImage<T>(floatingPyramid[l], referenceThresholdLow[0], referenceThresholdUp[0]);
    }

    initialised = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::Initialise");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
double reg_base<T>::ComputeSimilarityMeasure() {
    double measure = 0;
    if (measure_nmi != nullptr)
        measure += measure_nmi->GetSimilarityMeasureValue();

    if (measure_ssd != nullptr)
        measure += measure_ssd->GetSimilarityMeasureValue();

    if (measure_kld != nullptr)
        measure += measure_kld->GetSimilarityMeasureValue();

    if (measure_lncc != nullptr)
        measure += measure_lncc->GetSimilarityMeasureValue();

    if (measure_dti != nullptr)
        measure += measure_dti->GetSimilarityMeasureValue();

    if (measure_mind != nullptr)
        measure += measure_mind->GetSimilarityMeasureValue();

    if (measure_mindssc != nullptr)
        measure += measure_mindssc->GetSimilarityMeasureValue();

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ComputeSimilarityMeasure");
#endif
    return similarityWeight * measure;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::GetVoxelBasedGradient() {
    // The voxel based gradient image is filled with zeros
    // TODO Temporarily call F3dContent. This function will be moved to reg_f3d.
    dynamic_cast<F3dContent*>(con)->ZeroVoxelBasedMeasureGradient();

    // The intensity gradient is first computed
    //   if(measure_nmi!=nullptr || measure_ssd!=nullptr ||
    //         measure_kld!=nullptr || measure_lncc!=nullptr ||
    //         measure_dti!=nullptr)
    //   {
    //    if(measure_dti!=nullptr){
    //        reg_getImageGradient(floating,
    //                             warpedGradient,
    //                             deformationFieldImage,
    //                             currentMask,
    //                             interpolation,
    //                             warpedPaddingValue,
    //                             measure_dti->GetActiveTimepoints(),
    //		 					   forwardJacobianMatrix,
    //							   warped);
    //    }
    //    else{
    //    }
    //   }

    //   if(measure_dti!=nullptr)
    //      measure_dti->GetVoxelBasedSimilarityMeasureGradient();

    for (int t = 0; t < con->Content::GetReference()->nt; ++t) {
        compute->GetImageGradient(interpolation, warpedPaddingValue, t);

        // The gradient of the various measures of similarity are computed
        if (measure_nmi != nullptr)
            measure_nmi->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_ssd != nullptr)
            measure_ssd->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_kld != nullptr)
            measure_kld->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_lncc != nullptr)
            measure_lncc->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_mind != nullptr)
            measure_mind->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_mindssc != nullptr)
            measure_mindssc->GetVoxelBasedSimilarityMeasureGradient(t);
    }

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::GetVoxelBasedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
//template<class T>
//void reg_base<T>::ApproximateParzenWindow()
//{
//    if(measure_nmi==nullptr)
//        measure_nmi=new reg_nmi;
//    measure_nmi=approxParzenWindow = true;
//}
///* *************************************************************** */
//template<class T>
//void reg_base<T>::DoNotApproximateParzenWindow()
//{
//    if(measure_nmi==nullptr)
//        measure_nmi=new reg_nmi;
//    measure_nmi=approxParzenWindow = false;
//}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber) {
    if (measure_nmi == nullptr)
        measure_nmi = new reg_nmi;
    measure_nmi->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accommodate for
    // the spline support
    measure_nmi->SetReferenceBinNumber(refBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNMISetReferenceBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber) {
    if (measure_nmi == nullptr)
        measure_nmi = new reg_nmi;
    measure_nmi->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accommodate for
    // the spline support
    measure_nmi->SetFloatingBinNumber(floBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNMISetFloatingBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseSSD(int timepoint, bool normalise) {
    if (measure_ssd == nullptr)
        measure_ssd = new reg_ssd();
    measure_ssd->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    measure_ssd->SetNormaliseTimepoint(timepoint, normalise);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseSSD");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMIND(int timepoint, int offset) {
    if (measure_mind == nullptr)
        measure_mind = new reg_mind;
    measure_mind->SetTimepointWeight(timepoint, 1.0);//weight set to 1.0 to indicate timepoint is active
    measure_mind->SetDescriptorOffset(offset);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseMIND");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMINDSSC(int timepoint, int offset) {
    if (measure_mindssc == nullptr)
        measure_mindssc = new reg_mindssc;
    measure_mindssc->SetTimepointWeight(timepoint, 1.0);//weight set to 1.0 to indicate timepoint is active
    measure_mindssc->SetDescriptorOffset(offset);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseMINDSSC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseKLDivergence(int timepoint) {
    if (measure_kld == nullptr)
        measure_kld = new reg_kld;
    measure_kld->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseKLDivergence");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLNCC(int timepoint, float stddev) {
    if (measure_lncc == nullptr)
        measure_lncc = new reg_lncc;
    measure_lncc->SetKernelStandardDeviation(timepoint, stddev);
    measure_lncc->SetTimepointWeight(timepoint, 1.0); // weight initially set to default value of 1.0
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseLNCC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCKernelType(int type) {
    if (measure_lncc == nullptr) {
        reg_print_fct_error("reg_base<T>::SetLNCCKernelType");
        reg_print_msg_error("The LNCC object has to be created first");
        reg_exit();
    }
    measure_lncc->SetKernelType(type);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLNCCKernelType");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseDTI(bool *timepoint) {
    reg_print_msg_error("The use of DTI has been deactivated as it requires some refactoring");
    reg_exit();

    if (measure_dti == nullptr)
        measure_dti = new reg_dti;
    for (int i = 0; i < inputReference->nt; ++i) {
        if (timepoint[i])
            measure_dti->SetTimepointWeight(i, 1.0);  // weight set to 1.0 to indicate timepoint is active
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseDTI");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetNMIWeight(int timepoint, double weight) {
    if (measure_nmi == nullptr) {
        reg_print_fct_error("reg_base<T>::SetNMIWeight");
        reg_print_msg_error("The NMI object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    measure_nmi->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCWeight(int timepoint, double weight) {
    if (measure_lncc == nullptr) {
        reg_print_fct_error("reg_base<T>::SetLNCCWeight");
        reg_print_msg_error("The LNCC object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    measure_lncc->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetSSDWeight(int timepoint, double weight) {
    if (measure_ssd == nullptr) {
        reg_print_fct_error("reg_base<T>::SetSSDWeight");
        reg_print_msg_error("The SSD object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    measure_ssd->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetKLDWeight(int timepoint, double weight) {
    if (measure_kld == nullptr) {
        reg_print_fct_error("reg_base<T>::SetKLDWeight");
        reg_print_msg_error("The KLD object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    measure_kld->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLocalWeightSim(nifti_image *i) {
    localWeightSimInput = i;
    reg_tools_changeDatatype<T>(localWeightSimInput);
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::WarpFloatingImage(int inter) {
    // Compute the deformation field
    GetDeformationField();

    if (measure_dti == nullptr) {
        // Resample the floating image
        compute->ResampleImage(inter, warpedPaddingValue);
    } else {
        // reg_defField_getJacobianMatrix(deformationFieldImage, forwardJacobianMatrix);
        /*DTI needs fixing!
       reg_resampleImage(floating,
                          warped,
                          deformationFieldImage,
                          currentMask,
                          inter,
                          warpedPaddingValue,
                          measure_dti->GetActiveTimepoints(),
                          forwardJacobianMatrix);*/
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::WarpFloatingImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::Run() {
#ifndef NDEBUG
    char text[255];
    sprintf(text, "%s::Run() called", executableName);
    reg_print_msg_debug(text);
#endif

    Initialise();
#ifdef NDEBUG
    if (verbose) {
#endif
        reg_print_info(executableName, "***********************************************************");
#ifdef NDEBUG
    }
#endif

    // Update the maximal number of iteration to perform per level
    maxIterationNumber = maxIterationNumber * pow(2, levelToPerform - 1);

    // Loop over the different resolution level to perform
    for (currentLevel = 0; currentLevel < levelToPerform; currentLevel++) {
        // Set the current input images
        nifti_image *reference;
        nifti_image *floating;
        int *mask;
        if (usePyramid) {
            reference = referencePyramid[currentLevel];
            floating = floatingPyramid[currentLevel];
            mask = maskPyramid[currentLevel];
        } else {
            reference = referencePyramid[0];
            floating = floatingPyramid[0];
            mask = maskPyramid[0];
        }

        // Allocate image that depends on the reference image
        // AllocateWarped();
        // AllocateDeformationField();
        // AllocateWarpedGradient();

        // The grid is refined if necessary
        T maxStepSize = InitialiseCurrentLevel(reference);
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / (T)100.0;

        // Allocate image that are required to compute the gradient
        // AllocateVoxelBasedMeasureGradient();
        // AllocateTransformationGradient();

        InitContent(reference, floating, mask);

        DisplayCurrentLevelParameters();

        // Initialise the measures of similarity
        InitialiseSimilarity();

        // Initialise the optimiser
        SetOptimiser();

        // Loop over the number of perturbation to do
        for (size_t perturbation = 0; perturbation <= perturbationNumber; ++perturbation) {
            // Evaluate the objective function value
            UpdateBestObjFunctionValue();
            PrintInitialObjFunctionValue();

            // Iterate until convergence or until the max number of iteration is reach
            while (true) {

                if (currentSize == 0)
                    break;

                if (optimiser->GetCurrentIterationNumber() >= optimiser->GetMaxIterationNumber()) {
                    reg_print_msg_warn("The current level reached the maximum number of iteration");
                    break;
                }

                // Compute the objective function gradient
                GetObjectiveFunctionGradient();

                // Normalise the gradient
                NormaliseGradient();

                // Initialise the line search initial step size
                currentSize = currentSize > maxStepSize ? maxStepSize : currentSize;

                // A line search is performed
                optimiser->Optimise(maxStepSize, smallestSize, currentSize);

                // Update the objective function variables and print some information
                PrintCurrentObjFunctionValue(currentSize);

            } // while
            if (perturbation < perturbationNumber) {

                optimiser->Perturbation(smallestSize);
                currentSize = maxStepSize;
#ifdef NDEBUG
                if (verbose) {
#endif
                    char text[255];
                    reg_print_info(executableName, "Perturbation Step - The number of iteration is reset to 0");
                    sprintf(text, "Perturbation Step - Every control point positions is altered by [-%g %g]",
                            smallestSize, smallestSize);
                    reg_print_info(executableName, text);

#ifdef NDEBUG
                }
#endif
            }
        } // perturbation loop

        // Final folding correction
        CorrectTransformation();

        // Some cleaning is performed
        delete optimiser;
        optimiser = nullptr;
        DeinitContent();
        // if (localWeightSimCurrent) {
        //     nifti_image_free(localWeightSimCurrent);
        //     localWeightSimCurrent = nullptr;
        // }
        // DeallocateCurrentInputImage();
        // DeallocateWarped();
        // DeallocateDeformationField();
        // DeallocateWarpedGradient();
        // DeallocateVoxelBasedMeasureGradient();
        // DeallocateTransformationGradient();
        if (usePyramid) {
            nifti_image_free(referencePyramid[currentLevel]);
            referencePyramid[currentLevel] = nullptr;
            nifti_image_free(floatingPyramid[currentLevel]);
            floatingPyramid[currentLevel] = nullptr;
            free(maskPyramid[currentLevel]);
            maskPyramid[currentLevel] = nullptr;
        } else if (currentLevel == levelToPerform - 1) {
            nifti_image_free(referencePyramid[0]);
            referencePyramid[0] = nullptr;
            nifti_image_free(floatingPyramid[0]);
            floatingPyramid[0] = nullptr;
            free(maskPyramid[0]);
            maskPyramid[0] = nullptr;
        }

#ifdef NDEBUG
        if (verbose) {
#endif
            reg_print_info(executableName, "Current registration level done");
            reg_print_info(executableName, "***********************************************************");
#ifdef NDEBUG
        }
#endif
        // Update the number of level for the next level
        maxIterationNumber /= 2;
    } // level levelToPerform

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::Run");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template class reg_base<float>;
