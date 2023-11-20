#include "Content.h"
#include "_reg_tools.h"

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
    if (!referenceIn || !floatingIn)
        NR_FATAL_ERROR("referenceIn or floatingIn can't be nullptr");
    AllocateWarped();
    AllocateDeformationField(bytesIn);
    activeVoxelNumber = NiftiImage::calcVoxelNumber(reference, 3);
    if (!referenceMask) {
        referenceMaskManaged.reset(new int[activeVoxelNumber]());
        referenceMask = referenceMaskManaged.get();
    }
}
/* *************************************************************** */
Content::~Content() {
    DeallocateWarped();
    DeallocateDeformationField();
}
/* *************************************************************** */
void Content::AllocateWarped() {
    warped = nifti_copy_nim_info(reference);
    warped->dim[0] = warped->ndim = floating->ndim;
    warped->dim[4] = warped->nt = floating->nt;
    warped->pixdim[4] = warped->dt = 1;
    warped->nvox = NiftiImage::calcVoxelNumber(warped, warped->ndim);
    warped->datatype = floating->datatype;
    warped->nbyper = floating->nbyper;
    warped->data = calloc(warped->nvox, warped->nbyper);
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
    deformationField->dim[5] = deformationField->nu = reference->nz > 1 ? 3 : 2;
    deformationField->pixdim[5] = deformationField->du = 1;
    deformationField->dim[6] = deformationField->nv = 1;
    deformationField->pixdim[6] = deformationField->dv = 1;
    deformationField->dim[7] = deformationField->nw = 1;
    deformationField->pixdim[7] = deformationField->dw = 1;
    deformationField->nvox = NiftiImage::calcVoxelNumber(deformationField, deformationField->ndim);
    deformationField->nbyper = (int)bytes;
    if (bytes == 4)
        deformationField->datatype = NIFTI_TYPE_FLOAT32;
    else if (bytes == 8)
        deformationField->datatype = NIFTI_TYPE_FLOAT64;
    else
        NR_FATAL_ERROR("Only float or double are expected for the deformation field");
    deformationField->intent_code = NIFTI_INTENT_VECTOR;
    memset(deformationField->intent_name, 0, sizeof(deformationField->intent_name));
    strcpy(deformationField->intent_name, "NREG_TRANS");
    // First create a displacement field filled with 0 to obtain an identity disp
    deformationField->intent_p1 = DISP_FIELD;
    deformationField->scl_slope = 1;
    deformationField->scl_inter = 0;
    deformationField->data = calloc(deformationField->nvox, deformationField->nbyper);
    // Convert to an identity deformation field
    reg_getDeformationFromDisplacement(deformationField);
}
/* *************************************************************** */
void Content::DeallocateDeformationField() {
    if (deformationField) {
        nifti_image_free(deformationField);
        deformationField = nullptr;
    }
}
/* *************************************************************** */
