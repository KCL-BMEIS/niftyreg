#include "Content.h"
#include "_reg_maths.h"

/* *************************************************************** */
Content::Content(nifti_image *currentReferenceIn,
                 nifti_image *currentFloatingIn,
                 int *currentReferenceMaskIn,
                 mat44 *transformationMatrixIn,
                 size_t bytesIn) :
    currentReference(currentReferenceIn),
    currentFloating(currentFloatingIn),
    currentReferenceMask(currentReferenceMaskIn),
    transformationMatrix(transformationMatrixIn) {
    if (!currentReferenceIn || !currentFloatingIn) {
        reg_print_fct_error("Content::Content()");
        reg_print_msg_error("currentReferenceIn or currentFloatingIn can't be nullptr");
        reg_exit();
    }
    AllocateWarpedImage();
    AllocateDeformationField(bytesIn);
    if (currentReferenceMask == nullptr)
        currentReferenceMask = (int*)calloc(currentReference->nvox, sizeof(int));
}
/* *************************************************************** */
Content::~Content() {
    ClearWarpedImage();
    ClearDeformationField();
}
/* *************************************************************** */
void Content::AllocateWarpedImage() {
    currentWarped = nifti_copy_nim_info(currentReference);
    currentWarped->dim[0] = currentWarped->ndim = currentFloating->ndim;
    currentWarped->dim[4] = currentWarped->nt = currentFloating->nt;
    currentWarped->pixdim[4] = currentWarped->dt = 1.0;
    currentWarped->nvox = (size_t)(currentWarped->nx * currentWarped->ny * currentWarped->nz * currentWarped->nt);
    currentWarped->datatype = currentFloating->datatype;
    currentWarped->nbyper = currentFloating->nbyper;
    currentWarped->data = (void*)calloc(currentWarped->nvox, currentWarped->nbyper);
}
/* *************************************************************** */
void Content::ClearWarpedImage() {
    if (currentWarped)
        nifti_image_free(currentWarped);
    currentWarped = nullptr;
}
/* *************************************************************** */
void Content::AllocateDeformationField(size_t bytes) {
    currentDeformationField = nifti_copy_nim_info(currentReference);
    currentDeformationField->dim[0] = currentDeformationField->ndim = 5;
    if (currentReference->dim[0] == 2)
        currentDeformationField->dim[3] = currentDeformationField->nz = 1;
    currentDeformationField->dim[4] = currentDeformationField->nt = 1;
    currentDeformationField->pixdim[4] = currentDeformationField->dt = 1;
    if (currentReference->nz == 1)
        currentDeformationField->dim[5] = currentDeformationField->nu = 2;
    else
        currentDeformationField->dim[5] = currentDeformationField->nu = 3;
    currentDeformationField->pixdim[5] = currentDeformationField->du = 1;
    currentDeformationField->dim[6] = currentDeformationField->nv = 1;
    currentDeformationField->pixdim[6] = currentDeformationField->dv = 1;
    currentDeformationField->dim[7] = currentDeformationField->nw = 1;
    currentDeformationField->pixdim[7] = currentDeformationField->dw = 1;
    currentDeformationField->nvox = (size_t)(currentDeformationField->nx * currentDeformationField->ny * currentDeformationField->nz *
                                             currentDeformationField->nt * currentDeformationField->nu);
    currentDeformationField->nbyper = (int)bytes;
    if (bytes == 4)
        currentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
    else if (bytes == 8)
        currentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
    else {
        reg_print_fct_error("Content::AllocateDeformationField()");
        reg_print_msg_error("Only float or double are expected for the deformation field");
        reg_exit();
    }
    currentDeformationField->intent_code = NIFTI_INTENT_VECTOR;
    memset(currentDeformationField->intent_name, 0, sizeof(currentDeformationField->intent_name));
    strcpy(currentDeformationField->intent_name, "NREG_TRANS");
    currentDeformationField->intent_p1 = DEF_FIELD;
    currentDeformationField->scl_slope = 1;
    currentDeformationField->scl_inter = 0;
    currentDeformationField->data = (void*)calloc(currentDeformationField->nvox, currentDeformationField->nbyper);
}
/* *************************************************************** */
void Content::ClearDeformationField() {
    if (currentDeformationField)
        nifti_image_free(currentDeformationField);
    currentDeformationField = nullptr;
}
/* *************************************************************** */
