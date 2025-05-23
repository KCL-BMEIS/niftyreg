#include "Content.h"
#include "_reg_tools.h"

/* *************************************************************** */
Content::Content(NiftiImage& referenceIn,
                 NiftiImage& floatingIn,
                 int *referenceMaskIn,
                 mat44 *transformationMatrixIn,
                 size_t bytesIn):
    reference(NiftiImage(referenceIn, NiftiImage::Copy::Acquire)),
    floating(NiftiImage(floatingIn, NiftiImage::Copy::Acquire)),
    referenceMask(referenceMaskIn),
    transformationMatrix(transformationMatrixIn) {
    if (!referenceIn || !floatingIn)
        NR_FATAL_ERROR("referenceIn or floatingIn can't be nullptr");
    AllocateWarped();
    AllocateDeformationField(bytesIn);
    activeVoxelNumber = reference.nVoxelsPerVolume();
    if (!referenceMask) {
        referenceMaskManaged.reset(new int[activeVoxelNumber]());
        referenceMask = referenceMaskManaged.get();
    }
}
/* *************************************************************** */
void Content::AllocateWarped() {
    warped = NiftiImage(reference, NiftiImage::Copy::ImageInfo);
    warped.setDim(NiftiDim::NDim, floating->ndim);
    warped.setDim(NiftiDim::T, floating->nt);
    warped.setPixDim(NiftiDim::T, 1);
    warped->datatype = floating->datatype;
    warped->nbyper = floating->nbyper;
    warped.realloc();
}
/* *************************************************************** */
void Content::AllocateDeformationField(size_t bytes) {
    deformationField = NiftiImage(reference, NiftiImage::Copy::ImageInfo);
    deformationField.setDim(NiftiDim::NDim, 5);
    if (reference->ndim == 2)
        deformationField.setDim(NiftiDim::Z, 1);
    deformationField.setDim(NiftiDim::T, 1);
    deformationField.setPixDim(NiftiDim::T, 1);
    deformationField.setDim(NiftiDim::U, reference->nz > 1 ? 3 : 2);
    deformationField.setPixDim(NiftiDim::U, 1);
    deformationField.setDim(NiftiDim::V, 1);
    deformationField.setPixDim(NiftiDim::V, 1);
    deformationField.setDim(NiftiDim::W, 1);
    deformationField.setPixDim(NiftiDim::W, 1);
    deformationField->nbyper = (int)bytes;
    if (bytes == 4)
        deformationField->datatype = NIFTI_TYPE_FLOAT32;
    else if (bytes == 8)
        deformationField->datatype = NIFTI_TYPE_FLOAT64;
    else
        NR_FATAL_ERROR("Only float or double are expected for the deformation field");
    deformationField->intent_code = NIFTI_INTENT_VECTOR;
    deformationField.setIntentName("NREG_TRANS"s);
    // First create a displacement field filled with 0 to obtain an identity disp
    deformationField->intent_p1 = DISP_FIELD;
    deformationField->scl_slope = 1;
    deformationField->scl_inter = 0;
    deformationField.realloc();
    // Convert to an identity deformation field
    reg_getDeformationFromDisplacement(deformationField);
}
/* *************************************************************** */
