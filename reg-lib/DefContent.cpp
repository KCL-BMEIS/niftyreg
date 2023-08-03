#include "DefContent.h"
#include "_reg_resampling.h"

/* *************************************************************** */
DefContent::DefContent(nifti_image *referenceIn,
                       nifti_image *floatingIn,
                       nifti_image *localWeightSimIn,
                       int *referenceMaskIn,
                       mat44 *transformationMatrixIn,
                       size_t bytesIn):
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn) {
    AllocateWarpedGradient();
    AllocateVoxelBasedMeasureGradient();
    AllocateLocalWeightSim(localWeightSimIn);
}
/* *************************************************************** */
DefContent::~DefContent() {
    DeallocateWarpedGradient();
    DeallocateVoxelBasedMeasureGradient();
    DeallocateLocalWeightSim();
}
/* *************************************************************** */
void DefContent::AllocateLocalWeightSim(nifti_image *localWeightSimIn) {
    if (!localWeightSimIn) return;
    localWeightSim = nifti_copy_nim_info(reference);
    localWeightSim->dim[0] = localWeightSim->ndim = localWeightSimIn->dim[0];
    localWeightSim->dim[4] = localWeightSim->nt = localWeightSimIn->dim[4];
    localWeightSim->dim[5] = localWeightSim->nu = localWeightSimIn->dim[5];
    localWeightSim->nvox = NiftiImage::calcVoxelNumber(localWeightSim, localWeightSim->ndim);
    localWeightSim->data = malloc(localWeightSim->nvox * localWeightSim->nbyper);
    reg_getDeformationFromDisplacement(voxelBasedMeasureGradient);
    reg_resampleImage(localWeightSimIn, localWeightSim, voxelBasedMeasureGradient, nullptr, 1, 0);
}
/* *************************************************************** */
void DefContent::DeallocateLocalWeightSim() {
    if (localWeightSim) {
        nifti_image_free(localWeightSim);
        localWeightSim = nullptr;
    }
}
/* *************************************************************** */
void DefContent::AllocateWarpedGradient() {
    warpedGradient = nifti_dup(*deformationField, false);
}
/* *************************************************************** */
void DefContent::DeallocateWarpedGradient() {
    if (warpedGradient) {
        nifti_image_free(warpedGradient);
        warpedGradient = nullptr;
    }
}
/* *************************************************************** */
void DefContent::AllocateVoxelBasedMeasureGradient() {
    voxelBasedMeasureGradient = nifti_dup(*deformationField, false);
}
/* *************************************************************** */
void DefContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradient) {
        nifti_image_free(voxelBasedMeasureGradient);
        voxelBasedMeasureGradient = nullptr;
    }
}
/* *************************************************************** */
void DefContent::ZeroVoxelBasedMeasureGradient() {
    memset(voxelBasedMeasureGradient->data, 0, voxelBasedMeasureGradient->nvox * voxelBasedMeasureGradient->nbyper);
}
/* *************************************************************** */
