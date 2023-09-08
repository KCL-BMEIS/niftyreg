#include "_reg_aladin_sym.h"
#include "_reg_maths_eigen.h"

/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym()
    :reg_aladin<T>::reg_aladin() {
    this->executableName = (char*)"reg_aladin_sym";
    this->affineTransformationBw.reset(new mat44);
    this->backwardBlockMatchingParams = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(NiftiImage inputFloatingMaskIn) {
    this->inputFloatingMask = inputFloatingMaskIn;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration() {
    NR_FUNC_CALLED();

    reg_aladin<T>::InitialiseRegistration();

    this->floatingMaskPyramid = vector<unique_ptr<int[]>>(this->levelsToPerform);
    if (this->inputFloatingMask)
        reg_createMaskPyramid<T>(this->inputFloatingMask,
                                 this->floatingMaskPyramid,
                                 this->numberOfLevels,
                                 this->levelsToPerform);
    else
        for (unsigned l = 0; l < this->levelsToPerform; ++l)
            this->floatingMaskPyramid[l].reset(new int[this->floatingPyramid[l].nVoxelsPerVolume()]());

    // CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
    if (this->floatingUpperThreshold != std::numeric_limits<T>::max()) {
        for (unsigned l = 0; l < this->levelsToPerform; ++l) {
            T *refPtr = static_cast<T *>(this->floatingPyramid[l]->data);
            int *mskPtr = this->floatingMaskPyramid[l].get();
            for (size_t i = 0; i < this->floatingPyramid[l].nVoxelsPerVolume(); ++i) {
                if (mskPtr[i] > -1 && refPtr[i] > this->floatingUpperThreshold)
                    mskPtr[i] = -1;
            }
        }
    }
    if (this->floatingLowerThreshold != std::numeric_limits<T>::lowest()) {
        for (unsigned l = 0; l < this->levelsToPerform; ++l) {
            T *refPtr = static_cast<T *>(this->floatingPyramid[l]->data);
            int *mskPtr = this->floatingMaskPyramid[l].get();
            for (size_t i = 0; i < this->floatingPyramid[l].nVoxelsPerVolume(); ++i) {
                if (mskPtr[i] > -1 && refPtr[i] < this->floatingLowerThreshold)
                    mskPtr[i] = -1;
            }
        }
    }

    if (this->alignCentreMass == 1 && this->inputTransformName == nullptr) {
        if (!this->inputReferenceMask && !this->inputFloatingMask)
            NR_FATAL_ERROR("The masks' centre of mass can only be used when two masks are specified");

        float referenceCentre[3] = { 0, 0, 0 };
        float referenceCount = 0;
        reg_tools_changeDatatype<float>(this->inputReferenceMask);
        float *refMaskPtr = static_cast<float *>(this->inputReferenceMask->data);
        size_t refIndex = 0;
        for (int z = 0; z < this->inputReferenceMask->nz; ++z) {
            for (int y = 0; y < this->inputReferenceMask->ny; ++y) {
                for (int x = 0; x < this->inputReferenceMask->nx; ++x) {
                    if (refMaskPtr[refIndex] != 0.f) {
                        referenceCentre[0] += x;
                        referenceCentre[1] += y;
                        referenceCentre[2] += z;
                        referenceCount++;
                    }
                    refIndex++;
                }
            }
        }
        referenceCentre[0] /= referenceCount;
        referenceCentre[1] /= referenceCount;
        referenceCentre[2] /= referenceCount;
        float refCOG[3];
        if (this->inputReference->sform_code > 0)
            reg_mat44_mul(&(this->inputReference->sto_xyz), referenceCentre, refCOG);

        float floatingCentre[3] = { 0, 0, 0 };
        float floatingCount = 0;
        reg_tools_changeDatatype<float>(this->inputFloatingMask);
        float *floMaskPtr = static_cast<float *>(this->inputFloatingMask->data);
        size_t floIndex = 0;
        for (int z = 0; z < this->inputFloatingMask->nz; ++z) {
            for (int y = 0; y < this->inputFloatingMask->ny; ++y) {
                for (int x = 0; x < this->inputFloatingMask->nx; ++x) {
                    if (floMaskPtr[floIndex] != 0.f) {
                        floatingCentre[0] += x;
                        floatingCentre[1] += y;
                        floatingCentre[2] += z;
                        floatingCount++;
                    }
                    floIndex++;
                }
            }
        }
        floatingCentre[0] /= floatingCount;
        floatingCentre[1] /= floatingCount;
        floatingCentre[2] /= floatingCount;
        float floCOG[3];
        if (this->inputFloating->sform_code > 0)
            reg_mat44_mul(&(this->inputFloating->sto_xyz), floatingCentre, floCOG);
        reg_mat44_eye(this->affineTransformation.get());
        this->affineTransformation->m[0][3] = floCOG[0] - refCOG[0];
        this->affineTransformation->m[1][3] = floCOG[1] - refCOG[1];
        this->affineTransformation->m[2][3] = floCOG[2] - refCOG[2];
    }
    *this->affineTransformationBw = nifti_mat44_inverse(*this->affineTransformation);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField() {
    this->bAffineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->Calculate();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp, float padding) {
    reg_aladin<T>::GetWarpedImage(interp, padding);
    this->GetBackwardDeformationField();
    this->bResamplingKernel->template castTo<ResampleImageKernel>()->Calculate(interp, padding);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type) {
    reg_aladin<T>::UpdateTransformationMatrix(type);

    // Update now the backward transformation matrix
    this->bBlockMatchingKernel->template castTo<BlockMatchingKernel>()->Calculate();
    this->bLtsKernel->template castTo<LtsKernel>()->Calculate(type);

    NR_MAT44_DEBUG(*this->affineTransformation, "The pre-updated forward transformation matrix");
    NR_MAT44_DEBUG(*this->affineTransformationBw, "The pre-updated backward transformation matrix");

    // Forward and backward matrix are inverted
    mat44 fInverted = nifti_mat44_inverse(*this->affineTransformation);
    mat44 bInverted = nifti_mat44_inverse(*this->affineTransformationBw);

    // We average the forward and inverted backward matrix
    *this->affineTransformation = reg_mat44_avg2(this->affineTransformation.get(), &bInverted);
    // We average the inverted forward and backward matrix
    *this->affineTransformationBw = reg_mat44_avg2(&fInverted, this->affineTransformationBw.get());
    for (int i = 0; i < 3; ++i) {
        this->affineTransformation->m[3][i] = 0.f;
        this->affineTransformationBw->m[3][i] = 0.f;
    }
    this->affineTransformation->m[3][3] = 1.f;
    this->affineTransformationBw->m[3][3] = 1.f;

    NR_MAT44_DEBUG(*this->affineTransformation, "The updated forward transformation matrix");
    NR_MAT44_DEBUG(*this->affineTransformationBw, "The updated backward transformation matrix");
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitAladinContent(nifti_image *ref,
                                          nifti_image *flo,
                                          int *mask,
                                          mat44 *transMat,
                                          size_t bytes,
                                          unsigned blockPercentage,
                                          unsigned inlierLts,
                                          unsigned blockStepSize) {
    reg_aladin<T>::InitAladinContent(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize);
    unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(this->platform->CreateContentCreator(ContentType::Aladin)) };
    this->backCon.reset(contentCreator->Create(flo, ref, this->floatingMaskPyramid[this->currentLevel].get(), this->affineTransformationBw.get(), bytes, blockPercentage, inlierLts, blockStepSize));
    this->backwardBlockMatchingParams = backCon->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeallocateCurrentInputImage() {
    reg_aladin<T>::DeallocateCurrentInputImage();
    this->floatingMaskPyramid[this->currentLevel] = nullptr;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::CreateKernels() {
    reg_aladin<T>::CreateKernels();
    this->bAffineTransformation3DKernel.reset(this->platform->CreateKernel(AffineDeformationFieldKernel::GetName(), this->backCon.get()));
    this->bBlockMatchingKernel.reset(this->platform->CreateKernel(BlockMatchingKernel::GetName(), this->backCon.get()));
    this->bResamplingKernel.reset(this->platform->CreateKernel(ResampleImageKernel::GetName(), this->backCon.get()));
    this->bLtsKernel.reset(this->platform->CreateKernel(LtsKernel::GetName(), this->backCon.get()));
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeinitAladinContent() {
    reg_aladin<T>::DeinitAladinContent();
    this->backCon = nullptr;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeallocateKernels() {
    reg_aladin<T>::DeallocateKernels();
    this->bResamplingKernel = nullptr;
    this->bAffineTransformation3DKernel = nullptr;
    this->bBlockMatchingKernel = nullptr;
    this->bLtsKernel = nullptr;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart() {
    const nifti_image *ref = this->con->Content::GetReference();
    const nifti_image *flo = this->con->Content::GetFloating();
    NR_VERBOSE("Current level " << this->currentLevel + 1 << " / " << this->numberOfLevels);
    NR_VERBOSE("Reference image size:\t" << ref->nx << "x" << ref->ny << "x" << ref->nz << " voxels\t" <<
               ref->dx << "x" << ref->dy << "x" << ref->dz << " mm");
    NR_VERBOSE("Floating image size:\t" << flo->nx << "x" << flo->ny << "x" << flo->nz << " voxels\t" <<
               flo->dx << "x" << flo->dy << "x" << flo->dz << " mm");
    NR_VERBOSE("Block size = [4 4 " << (ref->nz == 1 ? 1 : 4) << "]");
    NR_VERBOSE("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_VERBOSE("Forward Block number = [" << this->blockMatchingParams->blockNumber[0] << " " <<
               this->blockMatchingParams->blockNumber[1] << " " << this->blockMatchingParams->blockNumber[2] << "]");
    NR_VERBOSE("Backward Block number = [" << this->backwardBlockMatchingParams->blockNumber[0] << " " <<
               this->backwardBlockMatchingParams->blockNumber[1] << " " << this->backwardBlockMatchingParams->blockNumber[2] << "]");
    NR_MAT44_VERBOSE(*this->affineTransformation, "Initial forward transformation matrix:");
    NR_MAT44_VERBOSE(*this->affineTransformationBw, "Initial backward transformation matrix:");
    NR_VERBOSE("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd() {
    NR_MAT44_VERBOSE(*this->affineTransformation, "Final forward transformation matrix:");
    NR_MAT44_VERBOSE(*this->affineTransformationBw, "Final backward transformation matrix:");
}
/* *************************************************************** */
template class reg_aladin_sym<float>;
