#include "Content.h"

/* *************************************************************** */
Content::Content(nifti_image *referenceIn,
                 nifti_image *floatingIn,
                 int *referenceMaskIn,
                 mat44 *transformationMatrixIn,
                 size_t bytesIn):
    reference(referenceIn),
    floating(floatingIn),
    referenceMask(referenceMaskIn),
    transformationMatrix(transformationMatrixIn) {
    if (!referenceIn || !floatingIn) {
        reg_print_fct_error("Content::Content()");
        reg_print_msg_error("referenceIn or floatingIn can't be nullptr");
        reg_exit();
    }
    AllocateWarped();
    AllocateDeformationField(bytesIn);
    if (!referenceMask)
        referenceMask = (int*)calloc(reference->nvox, sizeof(int));
}
/* *************************************************************** */
Content::~Content() {
    DeallocateWarped();
    DeallocateDeformationField();
    // free(referenceMask); // TODO Fix this with smart pointers
}
/* *************************************************************** */
void Content::AllocateWarped() {
    warped = nifti_copy_nim_info(reference);
    warped->dim[0] = warped->ndim = floating->ndim;
    warped->dim[4] = warped->nt = floating->nt;
    warped->pixdim[4] = warped->dt = 1.0;
    warped->nvox = size_t(warped->nx * warped->ny * warped->nz * warped->nt);
    warped->datatype = floating->datatype;
    warped->nbyper = floating->nbyper;
    warped->data = (void*)calloc(warped->nvox, warped->nbyper);
}
/* *************************************************************** */
void Content::DeallocateWarped() {
    if (warped) {
        nifti_image_free(warped);
        warped = nullptr;
    }
}
/* *************************************************************** */
void Content::AllocateDeformationField(size_t bytes) {
    deformationField = nifti_copy_nim_info(reference);
    deformationField->dim[0] = deformationField->ndim = 5;
    if (reference->dim[0] == 2)
        deformationField->dim[3] = deformationField->nz = 1;
    deformationField->dim[4] = deformationField->nt = 1;
    deformationField->pixdim[4] = deformationField->dt = 1;
    if (reference->nz == 1)
        deformationField->dim[5] = deformationField->nu = 2;
    else
        deformationField->dim[5] = deformationField->nu = 3;
    deformationField->pixdim[5] = deformationField->du = 1;
    deformationField->dim[6] = deformationField->nv = 1;
    deformationField->pixdim[6] = deformationField->dv = 1;
    deformationField->dim[7] = deformationField->nw = 1;
    deformationField->pixdim[7] = deformationField->dw = 1;
    deformationField->nvox = size_t(deformationField->nx * deformationField->ny * deformationField->nz *
                                    deformationField->nt * deformationField->nu);
    deformationField->nbyper = (int)bytes;
    if (bytes == 4)
        deformationField->datatype = NIFTI_TYPE_FLOAT32;
    else if (bytes == 8)
        deformationField->datatype = NIFTI_TYPE_FLOAT64;
    else {
        reg_print_fct_error("Content::AllocateDeformationField()");
        reg_print_msg_error("Only float or double are expected for the deformation field");
        reg_exit();
    }
    deformationField->intent_code = NIFTI_INTENT_VECTOR;
    memset(deformationField->intent_name, 0, sizeof(deformationField->intent_name));
    strcpy(deformationField->intent_name, "NREG_TRANS");
    deformationField->intent_p1 = DEF_FIELD;
    deformationField->scl_slope = 1;
    deformationField->scl_inter = 0;
    deformationField->data = (void*)calloc(deformationField->nvox, deformationField->nbyper);
}
/* *************************************************************** */
void Content::DeallocateDeformationField() {
    if (deformationField) {
        nifti_image_free(deformationField);
        deformationField = nullptr;
    }
}
/* *************************************************************** */
