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

 /* *************************************************************** */
 /* *************************************************************** */
template <class T>
   //Platform
//   this->platform = nullptr;
//   this->platformCode = NR_PLATFORM_CPU;
//   this->gpuIdx = 999;
reg_base<T>::reg_base(int refTimePoint, int floTimePoint) {

    this->optimiser = nullptr;
    this->maxIterationNumber = 150;
    this->optimiseX = true;
    this->optimiseY = true;
    this->optimiseZ = true;
    this->perturbationNumber = 0;
    this->useConjGradient = true;
    this->useApproxGradient = false;

    this->measure_ssd = nullptr;
    this->measure_kld = nullptr;
    this->measure_dti = nullptr;
    this->measure_lncc = nullptr;
    this->measure_nmi = nullptr;
    this->measure_mind = nullptr;
    this->measure_mindssc = nullptr;
    this->localWeightSimInput = nullptr;
    this->localWeightSimCurrent = nullptr;

    this->similarityWeight = 0; // automatically set depending of the penalty term weights

    this->executableName = (char*)"NiftyReg BASE";
    this->referenceTimePoint = refTimePoint;
    this->floatingTimePoint = floTimePoint;
    this->inputReference = nullptr; // pointer to external
    this->inputFloating = nullptr; // pointer to external
    this->maskImage = nullptr; // pointer to external
    this->affineTransformation = nullptr;  // pointer to external
    this->referenceMask = nullptr;
    this->referenceSmoothingSigma = 0;
    this->floatingSmoothingSigma = 0;
    this->referenceThresholdUp = new float[this->referenceTimePoint];
    this->referenceThresholdLow = new float[this->referenceTimePoint];
    this->floatingThresholdUp = new float[this->floatingTimePoint];
    this->floatingThresholdLow = new float[this->floatingTimePoint];
    for (int i = 0; i < this->referenceTimePoint; i++) {
        this->referenceThresholdUp[i] = std::numeric_limits<T>::max();
        this->referenceThresholdLow[i] = -std::numeric_limits<T>::max();
    }
    for (int i = 0; i < this->floatingTimePoint; i++) {
        this->floatingThresholdUp[i] = std::numeric_limits<T>::max();
        this->floatingThresholdLow[i] = -std::numeric_limits<T>::max();
    }
    this->robustRange = false;
    this->warpedPaddingValue = std::numeric_limits<T>::quiet_NaN();
    this->levelNumber = 3;
    this->levelToPerform = 0;
    this->gradientSmoothingSigma = 0;
    this->verbose = true;
    this->usePyramid = true;
    this->forwardJacobianMatrix = nullptr;

    this->initialised = false;
    this->referencePyramid = nullptr;
    this->floatingPyramid = nullptr;
    this->maskPyramid = nullptr;
    this->activeVoxelNumber = nullptr;
    this->currentReference = nullptr;
    this->currentFloating = nullptr;
    this->currentMask = nullptr;
    this->warped = nullptr;
    this->deformationFieldImage = nullptr;
    this->warImgGradient = nullptr;
    this->voxelBasedMeasureGradient = nullptr;

    this->interpolation = 1;

    this->landmarkRegWeight = 0;
    this->landmarkRegNumber = 0;
    this->landmarkReference = nullptr;
    this->landmarkFloating = nullptr;

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::reg_base");
#endif
}
/* *************************************************************** */
template <class T>
reg_base<T>::~reg_base() {
    this->ClearWarped();
    this->ClearWarpedGradient();
    this->ClearDeformationField();
    this->ClearVoxelBasedMeasureGradient();
    if (this->referencePyramid != nullptr) {
        if (this->usePyramid) {
            for (unsigned int i = 0; i < this->levelToPerform; i++) {
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
    if (this->maskPyramid != nullptr) {
        if (this->usePyramid) {
            for (unsigned int i = 0; i < this->levelToPerform; i++) {
                if (this->maskPyramid[i] != nullptr) {
                    free(this->maskPyramid[i]);
                    this->maskPyramid[i] = nullptr;
                }
            }
        } else {
            if (this->maskPyramid[0] != nullptr) {
                free(this->maskPyramid[0]);
                this->maskPyramid[0] = nullptr;
            }
        }
        free(this->maskPyramid);
        maskPyramid = nullptr;
    }
    if (this->floatingPyramid != nullptr) {
        if (this->usePyramid) {
            for (unsigned int i = 0; i < this->levelToPerform; i++) {
                if (floatingPyramid[i] != nullptr) {
                    nifti_image_free(floatingPyramid[i]);
                    floatingPyramid[i] = nullptr;
                }
            }
   //Platform
//   delete this->platform;
        } else {
            if (floatingPyramid[0] != nullptr) {
                nifti_image_free(floatingPyramid[0]);
                floatingPyramid[0] = nullptr;
            }
        }
        free(floatingPyramid);
        floatingPyramid = nullptr;
    }
    if (this->activeVoxelNumber != nullptr) {
        free(activeVoxelNumber);
        this->activeVoxelNumber = nullptr;
    }
    if (this->referenceThresholdUp != nullptr) {
        delete[]this->referenceThresholdUp;
        this->referenceThresholdUp = nullptr;
    }
    if (this->referenceThresholdLow != nullptr) {
        delete[]this->referenceThresholdLow;
        this->referenceThresholdLow = nullptr;
    }
    if (this->floatingThresholdUp != nullptr) {
        delete[]this->floatingThresholdUp;
        this->floatingThresholdUp = nullptr;
    }
    if (this->floatingThresholdLow != nullptr) {
        delete[]this->floatingThresholdLow;
        this->floatingThresholdLow = nullptr;
    }
    if (this->optimiser != nullptr) {
        delete this->optimiser;
        this->optimiser = nullptr;
    }

    if (this->measure_nmi != nullptr)
        delete this->measure_nmi;
    if (this->measure_ssd != nullptr)
        delete this->measure_ssd;
    if (this->measure_kld != nullptr)
        delete this->measure_kld;
    if (this->measure_dti != nullptr)
        delete this->measure_dti;
    if (this->measure_lncc != nullptr)
        delete this->measure_lncc;
    if (this->measure_mind != nullptr)
        delete this->measure_mind;
    if (this->measure_mindssc != nullptr)
        delete this->measure_mindssc;

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::~reg_base");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setPlaform(Platform* inputPlatform)
//{
//    this->platform = inputPlatform;
//}
/* *************************************************************** */
//template<class T>
//Platform* reg_base<T>::getPlaform()
//{
//    return this->platform;
//}
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setPlatformCode(int inputPlatformCode) {
//    this->platformCode = inputPlatformCode;
//}
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setGpuIdx(unsigned inputGPUIdx) {
//    this->gpuIdx = inputGPUIdx;
//}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceImage(nifti_image *r) {
    this->inputReference = r;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingImage(nifti_image *f) {
    this->inputFloating = f;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned int iter) {
    this->maxIterationNumber = iter;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetMaximalIterationNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceMask(nifti_image *m) {
    this->maskImage = m;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceMask");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetAffineTransformation(mat44 *a) {
    this->affineTransformation = a;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetAffineTransformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T s) {
    this->referenceSmoothingSigma = s;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T s) {
    this->floatingSmoothingSigma = s;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned int i, T t) {
    this->referenceThresholdUp[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned int i, T t) {
    this->referenceThresholdLow[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetReferenceThresholdLow");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned int i, T t) {
    this->floatingThresholdUp[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned int i, T t) {
    this->floatingThresholdLow[i] = t;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetFloatingThresholdLow");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::UseRobustRange() {
    this->robustRange = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::DoNotUseRobustRange() {
    this->robustRange = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(T p) {
    this->warpedPaddingValue = p;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetWarpedPaddingValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned int l) {
    this->levelNumber = l;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLevelNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned int l) {
    this->levelToPerform = l;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLevelToPerform");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T g) {
    this->gradientSmoothingSigma = g;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetGradientSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseConjugateGradient() {
    this->useConjGradient = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient() {
    this->useConjGradient = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseApproximatedGradient() {
    this->useApproxGradient = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient() {
    this->useApproxGradient = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::PrintOutInformation() {
    this->verbose = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::PrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotPrintOutInformation() {
    this->verbose = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotPrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach() {
    this->usePyramid = false;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::DoNotUsePyramidalApproach");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNearestNeighborInterpolation() {
    this->interpolation = 0;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNearestNeighborInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLinearInterpolation() {
    this->interpolation = 1;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseLinearInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation() {
    this->interpolation = 3;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseCubicSplineInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLandmarkRegularisationParam(size_t n, float *r, float *f, float w) {
    this->landmarkRegNumber = n;
    this->landmarkReference = r;
    this->landmarkFloating = f;
    this->landmarkRegWeight = w;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLandmarkRegularisationParam");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearCurrentInputImage() {
    this->currentReference = nullptr;
    this->currentMask = nullptr;
    this->currentFloating = nullptr;
    if (this->localWeightSimCurrent != nullptr)
        nifti_image_free(this->localWeightSimCurrent);
    this->localWeightSimCurrent = nullptr;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ClearCurrentInputImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::AllocateWarped() {
    if (this->currentReference == nullptr) {
        reg_print_fct_error("reg_base::AllocateWarped()");
        reg_print_msg_error("The reference image is not defined");
        reg_exit();
    }
    reg_base<T>::ClearWarped();
    this->warped = nifti_copy_nim_info(this->currentReference);
    this->warped->dim[0] = this->warped->ndim = this->currentFloating->ndim;
    this->warped->dim[4] = this->warped->nt = this->currentFloating->nt;
    this->warped->pixdim[4] = this->warped->dt = 1;
    this->warped->nvox = (size_t)(this->warped->nx * this->warped->ny * this->warped->nz * this->warped->nt);
    this->warped->scl_slope = 1;
    this->warped->scl_inter = 0;
    this->warped->datatype = this->currentFloating->datatype;
    this->warped->nbyper = this->currentFloating->nbyper;
    this->warped->data = (void*)calloc(this->warped->nvox, this->warped->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::AllocateWarped");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearWarped() {
    if (this->warped != nullptr)
        nifti_image_free(this->warped);
    this->warped = nullptr;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ClearWarped");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::AllocateDeformationField() {
    if (this->currentReference == nullptr) {
        reg_print_fct_error("reg_base::AllocateDeformationField()");
        reg_print_msg_error("The reference image is not defined");
        reg_exit();
    }
    reg_base<T>::ClearDeformationField();
    this->deformationFieldImage = nifti_copy_nim_info(this->currentReference);
    this->deformationFieldImage->dim[0] = this->deformationFieldImage->ndim = 5;
    this->deformationFieldImage->dim[1] = this->deformationFieldImage->nx = this->currentReference->nx;
    this->deformationFieldImage->dim[2] = this->deformationFieldImage->ny = this->currentReference->ny;
    this->deformationFieldImage->dim[3] = this->deformationFieldImage->nz = this->currentReference->nz;
    this->deformationFieldImage->dim[4] = this->deformationFieldImage->nt = 1;
    this->deformationFieldImage->pixdim[4] = this->deformationFieldImage->dt = 1.0;
    if (this->currentReference->nz == 1)
        this->deformationFieldImage->dim[5] = this->deformationFieldImage->nu = 2;
    else this->deformationFieldImage->dim[5] = this->deformationFieldImage->nu = 3;
    this->deformationFieldImage->pixdim[5] = this->deformationFieldImage->du = 1.0;
    this->deformationFieldImage->dim[6] = this->deformationFieldImage->nv = 1;
    this->deformationFieldImage->pixdim[6] = this->deformationFieldImage->dv = 1.0;
    this->deformationFieldImage->dim[7] = this->deformationFieldImage->nw = 1;
    this->deformationFieldImage->pixdim[7] = this->deformationFieldImage->dw = 1.0;
    this->deformationFieldImage->nvox =
        (size_t)this->deformationFieldImage->nx *
        (size_t)this->deformationFieldImage->ny *
        (size_t)this->deformationFieldImage->nz *
        (size_t)this->deformationFieldImage->nt *
        (size_t)this->deformationFieldImage->nu;
    this->deformationFieldImage->nbyper = sizeof(T);
    if (sizeof(T) == sizeof(float))
        this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
    else this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
    this->deformationFieldImage->data = (void*)calloc(this->deformationFieldImage->nvox,
                                                      this->deformationFieldImage->nbyper);
    this->deformationFieldImage->intent_code = NIFTI_INTENT_VECTOR;
    memset(this->deformationFieldImage->intent_name, 0, 16);
    strcpy(this->deformationFieldImage->intent_name, "NREG_TRANS");
    this->deformationFieldImage->intent_p1 = DEF_FIELD;
    this->deformationFieldImage->scl_slope = 1;
    this->deformationFieldImage->scl_inter = 0;

    if (this->measure_dti != nullptr)
        this->forwardJacobianMatrix = (mat33*)malloc(this->deformationFieldImage->nx * this->deformationFieldImage->ny *
                                                     this->deformationFieldImage->nz * sizeof(mat33));
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::AllocateDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearDeformationField() {
    if (this->deformationFieldImage != nullptr) {
        nifti_image_free(this->deformationFieldImage);
        this->deformationFieldImage = nullptr;
    }
    if (this->forwardJacobianMatrix != nullptr)
        free(this->forwardJacobianMatrix);
    this->forwardJacobianMatrix = nullptr;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ClearDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::AllocateWarpedGradient() {
    if (this->deformationFieldImage == nullptr) {
        reg_print_fct_error("reg_base::AllocateWarpedGradient()");
        reg_print_msg_error("The deformation field image is not defined");
        reg_exit();
    }
    reg_base<T>::ClearWarpedGradient();
    this->warImgGradient = nifti_copy_nim_info(this->deformationFieldImage);
    this->warImgGradient->data = (void*)calloc(this->warImgGradient->nvox,
                                                this->warImgGradient->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::AllocateWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearWarpedGradient() {
    if (this->warImgGradient != nullptr) {
        nifti_image_free(this->warImgGradient);
        this->warImgGradient = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ClearWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::AllocateVoxelBasedMeasureGradient() {
    if (this->deformationFieldImage == nullptr) {
        reg_print_fct_error("reg_base::AllocateVoxelBasedMeasureGradient()");
        reg_print_msg_error("The deformation field image is not defined");
        reg_exit();
    }
    reg_base<T>::ClearVoxelBasedMeasureGradient();
    this->voxelBasedMeasureGradient = nifti_copy_nim_info(this->deformationFieldImage);
    this->voxelBasedMeasureGradient->data = (void*)calloc(this->voxelBasedMeasureGradient->nvox,
                                                           this->voxelBasedMeasureGradient->nbyper);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::AllocateVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearVoxelBasedMeasureGradient() {
    if (this->voxelBasedMeasureGradient != nullptr) {
        nifti_image_free(this->voxelBasedMeasureGradient);
        this->voxelBasedMeasureGradient = nullptr;
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ClearVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::CheckParameters() {
    // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
    if (this->inputReference == nullptr) {
        reg_print_fct_error("reg_base::CheckParameters()");
        reg_print_msg_error("The reference image is not defined");
        reg_exit();
    }
    if (this->inputFloating == nullptr) {
        reg_print_fct_error("reg_base::CheckParameters()");
        reg_print_msg_error("The floating image is not defined");
        reg_exit();
    }

    // CHECK THE MASK DIMENSION IF IT IS DEFINED
    if (this->maskImage != nullptr) {
        if (this->inputReference->nx != this->maskImage->nx ||
            this->inputReference->ny != this->maskImage->ny ||
            this->inputReference->nz != this->maskImage->nz) {
            reg_print_fct_error("reg_base::CheckParameters()");
            reg_print_msg_error("The reference and mask images have different dimension");
            reg_exit();
        }
    }

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if (this->levelToPerform > 0) {
        this->levelToPerform = this->levelToPerform < this->levelNumber ? this->levelToPerform : this->levelNumber;
    } else this->levelToPerform = this->levelNumber;
    if (this->levelToPerform == 0 || this->levelToPerform > this->levelNumber)
        this->levelToPerform = this->levelNumber;

    // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
    if (this->measure_nmi == nullptr &&
        this->measure_ssd == nullptr &&
        this->measure_dti == nullptr &&
        this->measure_lncc == nullptr &&
        this->measure_lncc == nullptr &&
        this->measure_kld == nullptr &&
        this->measure_mind == nullptr &&
        this->measure_mindssc == nullptr) {
        this->measure_nmi = new reg_nmi;
        for (int i = 0; i < this->inputReference->nt; ++i)
            this->measure_nmi->SetTimepointWeight(i, 1.0);
    }

    // CHECK THAT IMAGES HAVE SAME NUMBER OF CHANNELS (TIMEPOINTS)
    // THAT EACH CHANNEL HAS AT LEAST ONE SIMILARITY MEASURE ASSIGNED
    // AND THAT EACH SIMILARITY MEASURE IS USED FOR AT LEAST ONE CHANNEL
    // NORMALISE CHANNEL AND SIMILARITY WEIGHTS SO TOTAL = 1
    //
    // NOTE - DTI currently ignored as needs fixing
    //
    // tests ignored if using MIND or MINDSSD as they are not implemented for multi-channel or weighting
    if (this->measure_mind == nullptr && this->measure_mindssc == nullptr) {
        if (this->inputFloating->nt != this->inputReference->nt) {
            reg_print_fct_error("reg_base::CheckParameters()");
            reg_print_msg_error("The reference and floating images have different numbers of channels (timepoints)");
            reg_exit();
        }
        double *chanWeightSum = new double[this->inputReference->nt]();
        double simWeightSum, totWeightSum = 0.;
        double *nmiWeights = nullptr, *ssdWeights = nullptr, *kldWeights = nullptr, *lnccWeights = nullptr;
        if (this->measure_nmi != nullptr) {
            nmiWeights = this->measure_nmi->GetTimepointsWeights();
            simWeightSum = 0.0;
            for (int n = 0; n < this->inputReference->nt; n++) {
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
            if (simWeightSum == 0.0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The NMI similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (this->measure_ssd != nullptr) {
            ssdWeights = this->measure_ssd->GetTimepointsWeights();
            simWeightSum = 0.0;
            for (int n = 0; n < this->inputReference->nt; n++) {
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
            if (simWeightSum == 0.0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The SSD similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (this->measure_kld != nullptr) {
            kldWeights = this->measure_kld->GetTimepointsWeights();
            simWeightSum = 0.0;
            for (int n = 0; n < this->inputReference->nt; n++) {
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
            if (simWeightSum == 0.0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The KLD similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        if (this->measure_lncc != nullptr) {
            lnccWeights = this->measure_lncc->GetTimepointsWeights();
            simWeightSum = 0.0;
            for (int n = 0; n < this->inputReference->nt; n++) {
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
            if (simWeightSum == 0.0) {
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn("The LNCC similarity measure has a weight of 0 for all channels so will be ignored");
            }
        }
        for (int n = 0; n < this->inputReference->nt; n++) {
            if (chanWeightSum[n] == 0) {
                char text[255];
                sprintf(text, "Channel %d has a weight of 0 for all similarity measures so will be ignored", n);
                reg_print_fct_warn("reg_base::CheckParameters()");
                reg_print_msg_warn(text);
            }
            if (this->measure_nmi != nullptr)
                this->measure_nmi->SetTimepointWeight(n, nmiWeights[n] / totWeightSum);
            if (this->measure_ssd != nullptr)
                this->measure_ssd->SetTimepointWeight(n, ssdWeights[n] / totWeightSum);
            if (this->measure_kld != nullptr)
                this->measure_kld->SetTimepointWeight(n, kldWeights[n] / totWeightSum);
            if (this->measure_lncc != nullptr)
                this->measure_lncc->SetTimepointWeight(n, lnccWeights[n] / totWeightSum);
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

    if (this->localWeightSimInput != nullptr) {
        if (this->localWeightSimCurrent != nullptr)
            nifti_image_free(this->localWeightSimCurrent);
        this->localWeightSimCurrent = nifti_copy_nim_info(this->currentReference);
        this->localWeightSimCurrent->dim[0] = this->localWeightSimCurrent->ndim = this->localWeightSimInput->dim[0];
        this->localWeightSimCurrent->dim[4] = this->localWeightSimCurrent->nt = this->localWeightSimInput->dim[4];
        this->localWeightSimCurrent->dim[5] = this->localWeightSimCurrent->nu = this->localWeightSimInput->dim[5];
        this->localWeightSimCurrent->nvox = (size_t)this->localWeightSimCurrent->nx *
            this->localWeightSimCurrent->ny * this->localWeightSimCurrent->nz *
            this->localWeightSimCurrent->nt * this->localWeightSimCurrent->nu;
        this->localWeightSimCurrent->data = (void*)malloc(this->localWeightSimCurrent->nvox *
                                                           this->localWeightSimCurrent->nbyper);
        reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradient, this->voxelBasedMeasureGradient, 0);
        reg_getDeformationFromDisplacement(this->voxelBasedMeasureGradient);
        reg_tools_changeDatatype<T>(localWeightSimInput);
        reg_resampleImage(this->localWeightSimInput,
                          this->localWeightSimCurrent,
                          this->voxelBasedMeasureGradient,
                          nullptr,
                          1,
                          0);
    } else this->localWeightSimCurrent = nullptr;

    if (this->measure_nmi != nullptr)
        this->measure_nmi->InitialiseMeasure(this->currentReference,
                                             this->currentFloating,
                                             this->currentMask,
                                             this->warped,
                                             this->warImgGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent);

    if (this->measure_ssd != nullptr)
        this->measure_ssd->InitialiseMeasure(this->currentReference,
                                             this->currentFloating,
                                             this->currentMask,
                                             this->warped,
                                             this->warImgGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent);

    if (this->measure_kld != nullptr)
        this->measure_kld->InitialiseMeasure(this->currentReference,
                                             this->currentFloating,
                                             this->currentMask,
                                             this->warped,
                                             this->warImgGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent);

    if (this->measure_lncc != nullptr)
        this->measure_lncc->InitialiseMeasure(this->currentReference,
                                              this->currentFloating,
                                              this->currentMask,
                                              this->warped,
                                              this->warImgGradient,
                                              this->voxelBasedMeasureGradient,
                                              this->localWeightSimCurrent);

    if (this->measure_dti != nullptr)
        this->measure_dti->InitialiseMeasure(this->currentReference,
                                             this->currentFloating,
                                             this->currentMask,
                                             this->warped,
                                             this->warImgGradient,
                                             this->voxelBasedMeasureGradient,
                                             this->localWeightSimCurrent);

    if (this->measure_mind != nullptr)
        this->measure_mind->InitialiseMeasure(this->currentReference,
                                              this->currentFloating,
                                              this->currentMask,
                                              this->warped,
                                              this->warImgGradient,
                                              this->voxelBasedMeasureGradient,
                                              this->localWeightSimCurrent);

    if (this->measure_mindssc != nullptr)
        this->measure_mindssc->InitialiseMeasure(this->currentReference,
                                                 this->currentFloating,
                                                 this->currentMask,
                                                 this->warped,
                                                 this->warImgGradient,
                                                 this->voxelBasedMeasureGradient,
                                                 this->localWeightSimCurrent);

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::InitialiseSimilarity");
#endif
}
/* *************************************************************** */
template<class T>
   //PLATFORM
//   this->platform = new Platform(this->platformCode);
//   this->platform->setGpuIdx(this->gpuIdx);
void reg_base<T>::Initialise() {
    if (this->initialised) return;

    this->CheckParameters();


    // CREATE THE PYRAMIDE IMAGES
    if (this->usePyramid) {
        this->referencePyramid = (nifti_image**)malloc(this->levelToPerform * sizeof(nifti_image*));
        this->floatingPyramid = (nifti_image**)malloc(this->levelToPerform * sizeof(nifti_image*));
        this->maskPyramid = (int**)malloc(this->levelToPerform * sizeof(int*));
        this->activeVoxelNumber = (int*)malloc(this->levelToPerform * sizeof(int));
    } else {
        this->referencePyramid = (nifti_image**)malloc(sizeof(nifti_image*));
        this->floatingPyramid = (nifti_image**)malloc(sizeof(nifti_image*));
        this->maskPyramid = (int**)malloc(sizeof(int*));
        this->activeVoxelNumber = (int*)malloc(sizeof(int));
    }

    // Update the input images threshold if required
    if (this->robustRange == true) {
        // Create a copy of the reference image to extract the robust range
        nifti_image *temp_reference = nifti_copy_nim_info(this->inputReference);
        temp_reference->data = (void*)malloc(temp_reference->nvox * temp_reference->nbyper);
        memcpy(temp_reference->data, this->inputReference->data, temp_reference->nvox * temp_reference->nbyper);
        reg_tools_changeDatatype<T>(temp_reference);
        // Extract the robust range of the reference image
        T *refDataPtr = static_cast<T *>(temp_reference->data);
        reg_heapSort(refDataPtr, temp_reference->nvox);
        // Update the reference threshold values if no value has been setup by the user
        if (this->referenceThresholdLow[0] == -std::numeric_limits<T>::max())
            this->referenceThresholdLow[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox * 0.02f)];
        if (this->referenceThresholdUp[0] == std::numeric_limits<T>::max())
            this->referenceThresholdUp[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox * 0.98f)];
        // Free the temporarly allocated image
        nifti_image_free(temp_reference);

        // Create a copy of the floating image to extract the robust range
        nifti_image *temp_floating = nifti_copy_nim_info(this->inputFloating);
        temp_floating->data = (void*)malloc(temp_floating->nvox * temp_floating->nbyper);
        memcpy(temp_floating->data, this->inputFloating->data, temp_floating->nvox * temp_floating->nbyper);
        reg_tools_changeDatatype<T>(temp_floating);
        // Extract the robust range of the floating image
        T *floDataPtr = static_cast<T *>(temp_floating->data);
        reg_heapSort(floDataPtr, temp_floating->nvox);
        // Update the floating threshold values if no value has been setup by the user
        if (this->floatingThresholdLow[0] == -std::numeric_limits<T>::max())
            this->floatingThresholdLow[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox * 0.02f)];
        if (this->floatingThresholdUp[0] == std::numeric_limits<T>::max())
            this->floatingThresholdUp[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox * 0.98f)];
        // Free the temporarly allocated image
        nifti_image_free(temp_floating);
    }

    // FINEST LEVEL OF REGISTRATION
    if (this->usePyramid) {
        reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, this->levelNumber, this->levelToPerform);
        reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, this->levelNumber, this->levelToPerform);
        if (this->maskImage != nullptr)
            reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, this->levelNumber, this->levelToPerform, this->activeVoxelNumber);
        else {
            for (unsigned int l = 0; l < this->levelToPerform; ++l) {
                this->activeVoxelNumber[l] = this->referencePyramid[l]->nx * this->referencePyramid[l]->ny * this->referencePyramid[l]->nz;
                this->maskPyramid[l] = (int*)calloc(activeVoxelNumber[l], sizeof(int));
            }
        }
    } else {
        reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, 1, 1);
        reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, 1, 1);
        if (this->maskImage != nullptr)
            reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, 1, 1, this->activeVoxelNumber);
        else {
            this->activeVoxelNumber[0] = this->referencePyramid[0]->nx * this->referencePyramid[0]->ny * this->referencePyramid[0]->nz;
            this->maskPyramid[0] = (int*)calloc(activeVoxelNumber[0], sizeof(int));
        }
    }

    unsigned int pyramidalLevelNumber = 1;
    if (this->usePyramid) pyramidalLevelNumber = this->levelToPerform;

    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < this->levelToPerform; l++) {
        if (this->referenceSmoothingSigma != 0.0) {
            bool *active = new bool[this->referencePyramid[l]->nt];
            float *sigma = new float[this->referencePyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < this->referencePyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->referenceSmoothingSigma;
            reg_tools_kernelConvolution(this->referencePyramid[l], sigma, GAUSSIAN_KERNEL, nullptr, active);
            delete[]active;
            delete[]sigma;
        }
        if (this->floatingSmoothingSigma != 0.0) {
            // Only the first image is smoothed
            bool *active = new bool[this->floatingPyramid[l]->nt];
            float *sigma = new float[this->floatingPyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < this->floatingPyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->floatingSmoothingSigma;
            reg_tools_kernelConvolution(this->floatingPyramid[l], sigma, GAUSSIAN_KERNEL, nullptr, active);
            delete[]active;
            delete[]sigma;
        }
    }

    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < pyramidalLevelNumber; l++) {
        reg_thresholdImage<T>(this->referencePyramid[l], this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
        reg_thresholdImage<T>(this->floatingPyramid[l], this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
    }

    this->initialised = true;
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::Initialise");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::SetOptimiser() {
    if (this->useConjGradient)
        this->optimiser = new reg_conjugateGradient<T>();
    else this->optimiser = new reg_optimiser<T>();
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_base<T>::ComputeSimilarityMeasure() {
    double measure = 0.;
    if (this->measure_nmi != nullptr)
        measure += this->measure_nmi->GetSimilarityMeasureValue();

    if (this->measure_ssd != nullptr)
        measure += this->measure_ssd->GetSimilarityMeasureValue();

    if (this->measure_kld != nullptr)
        measure += this->measure_kld->GetSimilarityMeasureValue();

    if (this->measure_lncc != nullptr)
        measure += this->measure_lncc->GetSimilarityMeasureValue();

    if (this->measure_dti != nullptr)
        measure += this->measure_dti->GetSimilarityMeasureValue();

    if (this->measure_mind != nullptr)
        measure += this->measure_mind->GetSimilarityMeasureValue();

    if (this->measure_mindssc != nullptr)
        measure += this->measure_mindssc->GetSimilarityMeasureValue();

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::ComputeSimilarityMeasure");
#endif
    return double(this->similarityWeight) * measure;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::GetVoxelBasedGradient() {
    // The voxel based gradient image is filled with zeros
    reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradient,
                                   this->voxelBasedMeasureGradient,
                                   0.f);

    // The intensity gradient is first computed
    //   if(this->measure_nmi!=nullptr || this->measure_ssd!=nullptr ||
    //         this->measure_kld!=nullptr || this->measure_lncc!=nullptr ||
    //         this->measure_dti!=nullptr)
    //   {
    //    if(this->measure_dti!=nullptr){
    //        reg_getImageGradient(this->currentFloating,
    //                             this->warImgGradient,
    //                             this->deformationFieldImage,
    //                             this->currentMask,
    //                             this->interpolation,
    //                             this->warpedPaddingValue,
    //                             this->measure_dti->GetActiveTimepoints(),
    //		 					   this->forwardJacobianMatrix,
    //							   this->warped);
    //    }
    //    else{
    //    }
    //   }

    //   if(this->measure_dti!=nullptr)
    //      this->measure_dti->GetVoxelBasedSimilarityMeasureGradient();

    for (int t = 0; t < this->currentReference->nt; ++t) {
        reg_getImageGradient(this->currentFloating,
                             this->warImgGradient,
                             this->deformationFieldImage,
                             this->currentMask,
                             this->interpolation,
                             this->warpedPaddingValue,
                             t);

        // The gradient of the various measures of similarity are computed
        if (this->measure_nmi != nullptr)
            this->measure_nmi->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_ssd != nullptr)
            this->measure_ssd->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_kld != nullptr)
            this->measure_kld->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_lncc != nullptr)
            this->measure_lncc->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_mind != nullptr)
            this->measure_mind->GetVoxelBasedSimilarityMeasureGradient(t);

        if (this->measure_mindssc != nullptr)
            this->measure_mindssc->GetVoxelBasedSimilarityMeasureGradient(t);
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
//    if(this->measure_nmi==nullptr)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = true;
//}
///* *************************************************************** */
//template<class T>
//void reg_base<T>::DoNotApproximateParzenWindow()
//{
//    if(this->measure_nmi==nullptr)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = false;
//}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber) {
    if (this->measure_nmi == nullptr)
        this->measure_nmi = new reg_nmi;
    this->measure_nmi->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accomodate for
    // the spline support
    this->measure_nmi->SetReferenceBinNumber(refBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNMISetReferenceBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber) {
    if (this->measure_nmi == nullptr)
        this->measure_nmi = new reg_nmi;
    this->measure_nmi->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accomodate for
    // the spline support
    this->measure_nmi->SetFloatingBinNumber(floBinNumber + 4, timepoint);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseNMISetFloatingBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseSSD(int timepoint, bool normalise) {
    if (this->measure_ssd == nullptr)
        this->measure_ssd = new reg_ssd();
    this->measure_ssd->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
    this->measure_ssd->SetNormaliseTimepoint(timepoint, normalise);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseSSD");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMIND(int timepoint, int offset) {
    if (this->measure_mind == nullptr)
        this->measure_mind = new reg_mind;
    this->measure_mind->SetTimepointWeight(timepoint, 1.0);//weight set to 1.0 to indicate timepoint is active
    this->measure_mind->SetDescriptorOffset(offset);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseMIND");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMINDSSC(int timepoint, int offset) {
    if (this->measure_mindssc == nullptr)
        this->measure_mindssc = new reg_mindssc;
    this->measure_mindssc->SetTimepointWeight(timepoint, 1.0);//weight set to 1.0 to indicate timepoint is active
    this->measure_mindssc->SetDescriptorOffset(offset);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseMINDSSC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseKLDivergence(int timepoint) {
    if (this->measure_kld == nullptr)
        this->measure_kld = new reg_kld;
    this->measure_kld->SetTimepointWeight(timepoint, 1.0);//weight initially set to default value of 1.0
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseKLDivergence");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLNCC(int timepoint, float stddev) {
    if (this->measure_lncc == nullptr)
        this->measure_lncc = new reg_lncc;
    this->measure_lncc->SetKernelStandardDeviation(timepoint, stddev);
    this->measure_lncc->SetTimepointWeight(timepoint, 1.0); // weight initially set to default value of 1.0
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseLNCC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCKernelType(int type) {
    if (this->measure_lncc == nullptr) {
        reg_print_fct_error("reg_base<T>::SetLNCCKernelType");
        reg_print_msg_error("The LNCC object has to be created first");
        reg_exit();
    }
    this->measure_lncc->SetKernelType(type);
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::SetLNCCKernelType");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseDTI(bool *timepoint) {
    reg_print_msg_error("The use of DTI has been deactivated as it requires some refactoring");
    reg_exit();

    if (this->measure_dti == nullptr)
        this->measure_dti = new reg_dti;
    for (int i = 0; i < this->inputReference->nt; ++i) {
        if (timepoint[i] == true)
            this->measure_dti->SetTimepointWeight(i, 1.0);  // weight set to 1.0 to indicate timepoint is active
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::UseDTI");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetNMIWeight(int timepoint, double weight) {
    if (this->measure_nmi == nullptr) {
        reg_print_fct_error("reg_base<T>::SetNMIWeight");
        reg_print_msg_error("The NMI object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    this->measure_nmi->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCWeight(int timepoint, double weight) {
    if (this->measure_lncc == nullptr) {
        reg_print_fct_error("reg_base<T>::SetLNCCWeight");
        reg_print_msg_error("The LNCC object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    this->measure_lncc->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetSSDWeight(int timepoint, double weight) {
    if (this->measure_ssd == nullptr) {
        reg_print_fct_error("reg_base<T>::SetSSDWeight");
        reg_print_msg_error("The SSD object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    this->measure_ssd->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetKLDWeight(int timepoint, double weight) {
    if (this->measure_kld == nullptr) {
        reg_print_fct_error("reg_base<T>::SetKLDWeight");
        reg_print_msg_error("The KLD object has to be created before the timepoint weights can be set");
        reg_exit();
    }
    this->measure_kld->SetTimepointWeight(timepoint, weight);
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLocalWeightSim(nifti_image *i) {
    this->localWeightSimInput = i;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::WarpFloatingImage(int inter) {
    // Compute the deformation field
    this->GetDeformationField();

    if (this->measure_dti == nullptr) {
        // Resample the floating image
        reg_resampleImage(this->currentFloating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          inter,
                          this->warpedPaddingValue);
    } else {
        reg_defField_getJacobianMatrix(this->deformationFieldImage,
                                       this->forwardJacobianMatrix);
        /*DTI needs fixing!
       reg_resampleImage(this->currentFloating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          inter,
                          this->warpedPaddingValue,
                          this->measure_dti->GetActiveTimepoints(),
                          this->forwardJacobianMatrix);*/
    }
#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::WarpFloatingImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::Run() {
#ifndef NDEBUG
    char text[255];
    sprintf(text, "%s::Run() called", this->executableName);
    reg_print_msg_debug(text);
#endif

    if (!this->initialised) this->Initialise();
#ifdef NDEBUG
    if (this->verbose) {
#endif
        reg_print_info(this->executableName, "***********************************************************");
#ifdef NDEBUG
    }
#endif

    // Update the maximal number of iteration to perform per level
    this->maxIterationNumber = this->maxIterationNumber * pow(2, this->levelToPerform - 1);

    // Loop over the different resolution level to perform
    for (this->currentLevel = 0;
         this->currentLevel < this->levelToPerform;
         this->currentLevel++) {

        // Set the current input images
        if (this->usePyramid) {
            this->currentReference = this->referencePyramid[this->currentLevel];
            this->currentFloating = this->floatingPyramid[this->currentLevel];
            this->currentMask = this->maskPyramid[this->currentLevel];
        } else {
            this->currentReference = this->referencePyramid[0];
            this->currentFloating = this->floatingPyramid[0];
            this->currentMask = this->maskPyramid[0];
        }

        // Allocate image that depends on the reference image
        this->AllocateWarped();
        this->AllocateDeformationField();
        this->AllocateWarpedGradient();

        // The grid is refined if necessary
        T maxStepSize = this->InitialiseCurrentLevel();
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / (T)100.0;

        this->DisplayCurrentLevelParameters();

        // Allocate image that are required to compute the gradient
        this->AllocateVoxelBasedMeasureGradient();
        this->AllocateTransformationGradient();

        // Initialise the measures of similarity
        this->InitialiseSimilarity();

        // initialise the optimiser
        this->SetOptimiser();

        // Loop over the number of perturbation to do
        for (size_t perturbation = 0;
             perturbation <= this->perturbationNumber;
             ++perturbation) {
            // Evalulate the objective function value
            this->UpdateBestObjFunctionValue();
            this->PrintInitialObjFunctionValue();

            // Iterate until convergence or until the max number of iteration is reach
            while (true) {

                if (currentSize == 0)
                    break;

                if (this->optimiser->GetCurrentIterationNumber() >= this->optimiser->GetMaxIterationNumber()) {
                    reg_print_msg_warn("The current level reached the maximum number of iteration");
                    break;
                }

                // Compute the objective function gradient
                this->GetObjectiveFunctionGradient();

                // Normalise the gradient
                this->NormaliseGradient();

                // Initialise the line search initial step size
                currentSize = currentSize > maxStepSize ? maxStepSize : currentSize;

                // A line search is performed
                this->optimiser->Optimise(maxStepSize, smallestSize, currentSize);

                // Update the obecjtive function variables and print some information
                this->PrintCurrentObjFunctionValue(currentSize);

            } // while
            if (perturbation < this->perturbationNumber) {

                this->optimiser->Perturbation(smallestSize);
                currentSize = maxStepSize;
#ifdef NDEBUG
                if (this->verbose) {
#endif
                    char text[255];
                    reg_print_info(this->executableName, "Perturbation Step - The number of iteration is reset to 0");
                    sprintf(text, "Perturbation Step - Every control point positions is altered by [-%g %g]",
                            smallestSize, smallestSize);
                    reg_print_info(this->executableName, text);

#ifdef NDEBUG
                }
#endif
            }
        } // perturbation loop

        // Final folding correction
        this->CorrectTransformation();

        // Some cleaning is performed
        delete this->optimiser;
        this->optimiser = nullptr;
        this->ClearWarped();
        this->ClearDeformationField();
        this->ClearWarpedGradient();
        this->ClearVoxelBasedMeasureGradient();
        this->ClearTransformationGradient();
        if (this->usePyramid) {
            nifti_image_free(this->referencePyramid[this->currentLevel]);
            this->referencePyramid[this->currentLevel] = nullptr;
            nifti_image_free(this->floatingPyramid[this->currentLevel]);
            this->floatingPyramid[this->currentLevel] = nullptr;
            free(this->maskPyramid[this->currentLevel]);
            this->maskPyramid[this->currentLevel] = nullptr;
        } else if (this->currentLevel == this->levelToPerform - 1) {
            nifti_image_free(this->referencePyramid[0]);
            this->referencePyramid[0] = nullptr;
            nifti_image_free(this->floatingPyramid[0]);
            this->floatingPyramid[0] = nullptr;
            free(this->maskPyramid[0]);
            this->maskPyramid[0] = nullptr;
        }
        this->ClearCurrentInputImage();

#ifdef NDEBUG
        if (this->verbose) {
#endif
            reg_print_info(this->executableName, "Current registration level done");
            reg_print_info(this->executableName, "***********************************************************");
#ifdef NDEBUG
        }
#endif
        // Update the number of level for the next level
        this->maxIterationNumber /= 2;
    } // level this->levelToPerform
    // Set this to the last value since it's used somewhere else
    this->currentLevel--;

#ifndef NDEBUG
    reg_print_fct_debug("reg_base<T>::Run");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template class reg_base<float>;
