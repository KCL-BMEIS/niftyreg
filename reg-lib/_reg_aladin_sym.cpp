#include "_reg_aladin_sym.h"
#include "_reg_maths_eigen.h"

/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym()
    :reg_aladin<T>::reg_aladin() {
    this->executableName = (char*)"reg_aladin_sym";

    this->backwardTransformationMatrix = new mat44;

    this->backwardBlockMatchingParams = nullptr;

#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin_sym constructor called");
#endif
}
/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym() {
    if (this->backwardTransformationMatrix)
        delete this->backwardTransformationMatrix;

#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin_sym destructor called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(NiftiImage inputFloatingMaskIn) {
    this->inputFloatingMask = inputFloatingMaskIn;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration() {
#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin_sym::InitialiseRegistration() called");
#endif

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
        if (!this->inputReferenceMask && !this->inputFloatingMask) {
            reg_print_msg_error("The masks' centre of mass can only be used when two masks are specified");
            reg_exit();
        }
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
        reg_mat44_eye(this->transformationMatrix);
        this->transformationMatrix->m[0][3] = floCOG[0] - refCOG[0];
        this->transformationMatrix->m[1][3] = floCOG[1] - refCOG[1];
        this->transformationMatrix->m[2][3] = floCOG[2] - refCOG[2];
    }
    *this->backwardTransformationMatrix = nifti_mat44_inverse(*this->transformationMatrix);
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
    this->bOptimiseKernel->template castTo<OptimiseKernel>()->Calculate(type);

#ifndef NDEBUG
    reg_mat44_disp(this->transformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated forward transformation matrix");
    reg_mat44_disp(this->backwardTransformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated backward transformation matrix");
#endif
    // Forward and backward matrix are inverted
    mat44 fInverted = nifti_mat44_inverse(*this->transformationMatrix);
    mat44 bInverted = nifti_mat44_inverse(*this->backwardTransformationMatrix);

    // We average the forward and inverted backward matrix
    *this->transformationMatrix = reg_mat44_avg2(this->transformationMatrix, &bInverted);
    // We average the inverted forward and backward matrix
    *this->backwardTransformationMatrix = reg_mat44_avg2(&fInverted, this->backwardTransformationMatrix);
    for (int i = 0; i < 3; ++i) {
        this->transformationMatrix->m[3][i] = 0.f;
        this->backwardTransformationMatrix->m[3][i] = 0.f;
    }
    this->transformationMatrix->m[3][3] = 1.f;
    this->backwardTransformationMatrix->m[3][3] = 1.f;
#ifndef NDEBUG
    reg_mat44_disp(this->transformationMatrix, (char *)"[NiftyReg DEBUG] updated forward transformation matrix");
    reg_mat44_disp(this->backwardTransformationMatrix, (char *)"[NiftyReg DEBUG] updated backward transformation matrix");
#endif
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
    this->backCon.reset(contentCreator->Create(flo, ref, this->floatingMaskPyramid[this->currentLevel].get(), this->backwardTransformationMatrix, bytes, blockPercentage, inlierLts, blockStepSize));
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
    this->bOptimiseKernel.reset(this->platform->CreateKernel(OptimiseKernel::GetName(), this->backCon.get()));
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
    this->bOptimiseKernel = nullptr;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart() {
    char text[255];
    sprintf(text, "Current level %i / %i", this->currentLevel + 1, this->numberOfLevels);
    reg_print_info(this->executableName, text);
    sprintf(text, "reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
            this->con->GetReference()->nx,
            this->con->GetReference()->ny,
            this->con->GetReference()->nz,
            this->con->GetReference()->dx,
            this->con->GetReference()->dy,
            this->con->GetReference()->dz);
    reg_print_info(this->executableName, text);
    sprintf(text, "floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
            this->con->GetFloating()->nx,
            this->con->GetFloating()->ny,
            this->con->GetFloating()->nz,
            this->con->GetFloating()->dx,
            this->con->GetFloating()->dy,
            this->con->GetFloating()->dz);
    reg_print_info(this->executableName, text);
    if (this->con->GetReference()->nz == 1) {
        reg_print_info(this->executableName, "Block size = [4 4 1]");
    } else reg_print_info(this->executableName, "Block size = [4 4 4]");
    reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    sprintf(text, "Forward Block number = [%i %i %i]", this->blockMatchingParams->blockNumber[0],
            this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
    reg_print_info(this->executableName, text);
    sprintf(text, "Backward Block number = [%i %i %i]", this->backwardBlockMatchingParams->blockNumber[0],
            this->backwardBlockMatchingParams->blockNumber[1], this->backwardBlockMatchingParams->blockNumber[2]);
    reg_print_info(this->executableName, text);
    reg_mat44_disp(this->transformationMatrix,
                   (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
    reg_mat44_disp(this->backwardTransformationMatrix,
                   (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
    reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd() {
    reg_mat44_disp(this->transformationMatrix, (char *)"[reg_aladin_sym] Final forward transformation matrix:");
    reg_mat44_disp(this->backwardTransformationMatrix, (char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* *************************************************************** */
template class reg_aladin_sym<float>;
