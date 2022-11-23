#include "AladinContent.h"

using namespace std;

/* *************************************************************** */
AladinContent::AladinContent() {
    //int dim[8] = { 2, 20, 20, 1, 1, 1, 1, 1 };
    //this->currentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    //this->currentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    //this->currentReferenceMask = nullptr;

    this->currentReference = nullptr;
    this->currentReferenceMask = nullptr;
    this->currentFloating = nullptr;
    this->transformationMatrix = nullptr;
    this->blockMatchingParams = nullptr;
    this->bytes = sizeof(float);  // Default

    InitVars();
}
/* *************************************************************** */
AladinContent::AladinContent(nifti_image *currentReferenceIn,
                             nifti_image *currentFloatingIn,
                             int *currentReferenceMaskIn,
                             mat44 *transMat,
                             size_t bytesIn,
                             const unsigned int currentPercentageOfBlockToUseIn,
                             const unsigned int inlierLtsIn,
                             int stepSizeBlockIn) :
    currentReference(currentReferenceIn),
    currentFloating(currentFloatingIn),
    currentReferenceMask(currentReferenceMaskIn),
    transformationMatrix(transMat),
    bytes(bytesIn),
    currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
    inlierLts(inlierLtsIn),
    stepSizeBlock(stepSizeBlockIn) {
    this->blockMatchingParams = new _reg_blockMatchingParam();
    InitVars();
}
/* *************************************************************** */
AladinContent::AladinContent(nifti_image *currentReferenceIn,
                             nifti_image *currentFloatingIn,
                             int *currentReferenceMaskIn,
                             mat44 *transMat,
                             size_t bytesIn) :
    currentReference(currentReferenceIn),
    currentFloating(currentFloatingIn),
    currentReferenceMask(currentReferenceMaskIn),
    transformationMatrix(transMat),
    bytes(bytesIn) {
    this->blockMatchingParams = nullptr;
    InitVars();
}
/* *************************************************************** */
AladinContent::AladinContent(nifti_image *currentReferenceIn,
                             nifti_image *currentFloatingIn,
                             int *currentReferenceMaskIn,
                             size_t bytesIn,
                             const unsigned int currentPercentageOfBlockToUseIn,
                             const unsigned int inlierLtsIn,
                             int stepSizeBlockIn) :
    currentReference(currentReferenceIn),
    currentFloating(currentFloatingIn),
    currentReferenceMask(currentReferenceMaskIn),
    bytes(bytesIn),
    currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
    inlierLts(inlierLtsIn),
    stepSizeBlock(stepSizeBlockIn) {
    this->transformationMatrix = nullptr;
    this->blockMatchingParams = new _reg_blockMatchingParam();
    InitVars();
}
/* *************************************************************** */
AladinContent::AladinContent(nifti_image *currentReferenceIn,
                             nifti_image *currentFloatingIn,
                             int *currentReferenceMaskIn,
                             size_t bytesIn) :
    currentReference(currentReferenceIn),
    currentFloating(currentFloatingIn),
    currentReferenceMask(currentReferenceMaskIn),
    bytes(bytesIn) {
    this->transformationMatrix = nullptr;
    this->blockMatchingParams = nullptr;
    InitVars();
}
/* *************************************************************** */
AladinContent::~AladinContent() {
    ClearWarpedImage();
    ClearDeformationField();
    if (this->blockMatchingParams != nullptr)
        delete this->blockMatchingParams;
}
/* *************************************************************** */
void AladinContent::InitVars() {
    if (this->currentFloating != nullptr && this->currentReference != nullptr) {
        this->AllocateWarpedImage();
    } else {
        this->currentWarped = nullptr;
    }

    if (this->currentReference != nullptr) {
        this->AllocateDeformationField(bytes);
        refMatrix_xyz = (currentReference->sform_code > 0) ? (currentReference->sto_xyz) : (currentReference->qto_xyz);
    } else {
        this->currentDeformationField = nullptr;
    }

    if (this->currentReferenceMask == nullptr && this->currentReference != nullptr)
        this->currentReferenceMask = (int *)calloc(this->currentReference->nx * this->currentReference->ny * this->currentReference->nz, sizeof(int));

    if (this->currentFloating != nullptr) {
        floMatrix_ijk = (currentFloating->sform_code > 0) ? (currentFloating->sto_ijk) : (currentFloating->qto_ijk);
    }
    if (blockMatchingParams != nullptr) {
        initialise_block_matching_method(currentReference,
                                         blockMatchingParams,
                                         currentPercentageOfBlockToUse,
                                         inlierLts,
                                         stepSizeBlock,
                                         currentReferenceMask,
                                         false);
    }
#ifndef NDEBUG
    if (this->currentReference == nullptr) reg_print_msg_debug("currentReference image is nullptr");
    if (this->currentFloating == nullptr) reg_print_msg_debug("currentFloating image is nullptr");
    if (this->currentDeformationField == nullptr) reg_print_msg_debug("currentDeformationField image is nullptr");
    if (this->currentWarped == nullptr) reg_print_msg_debug("currentWarped image is nullptr");
    if (this->currentReferenceMask == nullptr) reg_print_msg_debug("currentReferenceMask image is nullptr");
    if (this->blockMatchingParams == nullptr) reg_print_msg_debug("blockMatchingParams image is nullptr");
#endif
}
/* *************************************************************** */
void AladinContent::AllocateWarpedImage() {
    if (this->currentReference == nullptr || this->currentFloating == nullptr) {
        reg_print_fct_error("AladinContent::AllocateWarpedImage()");
        reg_print_msg_error(" Reference and floating images are not defined. Exit.");
        reg_exit();
    }

    this->currentWarped = nifti_copy_nim_info(this->currentReference);
    this->currentWarped->dim[0] = this->currentWarped->ndim = this->currentFloating->ndim;
    this->currentWarped->dim[4] = this->currentWarped->nt = this->currentFloating->nt;
    this->currentWarped->pixdim[4] = this->currentWarped->dt = 1.0;
    this->currentWarped->nvox = (size_t)(this->currentWarped->nx * this->currentWarped->ny * this->currentWarped->nz * this->currentWarped->nt);
    this->currentWarped->datatype = this->currentFloating->datatype;
    this->currentWarped->nbyper = this->currentFloating->nbyper;
    this->currentWarped->data = (void*)calloc(this->currentWarped->nvox, this->currentWarped->nbyper);
    //this->floatingDatatype = this->currentFloating->datatype;
}
/* *************************************************************** */
void AladinContent::AllocateDeformationField(size_t bytes) {
    if (this->currentReference == nullptr) {
        reg_print_fct_error("AladinContent::AllocateDeformationField()");
        reg_print_msg_error("Reference image is not defined. Exit.");
        reg_exit();
    }
    //ClearDeformationField();

    this->currentDeformationField = nifti_copy_nim_info(this->currentReference);
    this->currentDeformationField->dim[0] = this->currentDeformationField->ndim = 5;
    if (this->currentReference->dim[0] == 2)
        this->currentDeformationField->dim[3] = this->currentDeformationField->nz = 1;
    this->currentDeformationField->dim[4] = this->currentDeformationField->nt = 1;
    this->currentDeformationField->pixdim[4] = this->currentDeformationField->dt = 1.0;
    if (this->currentReference->nz == 1)
        this->currentDeformationField->dim[5] = this->currentDeformationField->nu = 2;
    else
        this->currentDeformationField->dim[5] = this->currentDeformationField->nu = 3;
    this->currentDeformationField->pixdim[5] = this->currentDeformationField->du = 1.0;
    this->currentDeformationField->dim[6] = this->currentDeformationField->nv = 1;
    this->currentDeformationField->pixdim[6] = this->currentDeformationField->dv = 1.0;
    this->currentDeformationField->dim[7] = this->currentDeformationField->nw = 1;
    this->currentDeformationField->pixdim[7] = this->currentDeformationField->dw = 1.0;
    this->currentDeformationField->nvox = (size_t)this->currentDeformationField->nx *
        this->currentDeformationField->ny * this->currentDeformationField->nz *
        this->currentDeformationField->nt * this->currentDeformationField->nu;
    this->currentDeformationField->nbyper = bytes;
    if (bytes == 4)
        this->currentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
    else if (bytes == 8)
        this->currentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
    else {
        reg_print_fct_error("AladinContent::AllocateDeformationField()");
        reg_print_msg_error("Only float or double are expected for the deformation field. Exit.");
        reg_exit();
    }
    this->currentDeformationField->scl_slope = 1;
    this->currentDeformationField->scl_inter = 0;
    this->currentDeformationField->data = (void*)calloc(this->currentDeformationField->nvox, this->currentDeformationField->nbyper);
}
/* *************************************************************** */
void AladinContent::SetCaptureRange(const int voxelCaptureRangeIn) {
    this->blockMatchingParams->voxelCaptureRange = voxelCaptureRangeIn;
}
/* *************************************************************** */
void AladinContent::ClearDeformationField() {
    if (this->currentDeformationField != nullptr)
        nifti_image_free(this->currentDeformationField);
    this->currentDeformationField = nullptr;
}
/* *************************************************************** */
void AladinContent::ClearWarpedImage() {
    if (this->currentWarped != nullptr)
        nifti_image_free(this->currentWarped);
    this->currentWarped = nullptr;
}
/* *************************************************************** */
bool AladinContent::IsCurrentComputationDoubleCapable() {
    return true;
}
