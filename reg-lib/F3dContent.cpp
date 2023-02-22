#include "F3dContent.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"

/* *************************************************************** */
F3dContent::F3dContent(nifti_image *referenceIn,
                       nifti_image *floatingIn,
                       nifti_image *controlPointGridIn,
                       nifti_image *localWeightSimIn,
                       int *referenceMaskIn,
                       mat44 *transformationMatrixIn,
                       size_t bytesIn):
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    controlPointGrid(controlPointGridIn) {
    if (!controlPointGridIn) {
        reg_print_fct_error("F3dContent::F3dContent()");
        reg_print_msg_error("controlPointGridIn can't be nullptr");
        reg_exit();
    }
    AllocateWarpedGradient();
    AllocateTransformationGradient();
    AllocateVoxelBasedMeasureGradient();
    AllocateLocalWeightSim(localWeightSimIn);
}
/* *************************************************************** */
F3dContent::~F3dContent() {
    DeallocateWarpedGradient();
    DeallocateTransformationGradient();
    DeallocateVoxelBasedMeasureGradient();
    DeallocateLocalWeightSim();
}
/* *************************************************************** */
void F3dContent::AllocateLocalWeightSim(nifti_image *localWeightSimIn) {
    if (!localWeightSimIn) return;
    localWeightSim = nifti_copy_nim_info(reference);
    localWeightSim->dim[0] = localWeightSim->ndim = localWeightSimIn->dim[0];
    localWeightSim->dim[4] = localWeightSim->nt = localWeightSimIn->dim[4];
    localWeightSim->dim[5] = localWeightSim->nu = localWeightSimIn->dim[5];
    localWeightSim->nvox = CalcVoxelNumber(*localWeightSim, localWeightSim->ndim);
    localWeightSim->data = malloc(localWeightSim->nvox * localWeightSim->nbyper);
    reg_getDeformationFromDisplacement(voxelBasedMeasureGradient);
    reg_resampleImage(localWeightSimIn, localWeightSim, voxelBasedMeasureGradient, nullptr, 1, 0);
}
/* *************************************************************** */
void F3dContent::DeallocateLocalWeightSim() {
    if (localWeightSim) {
        nifti_image_free(localWeightSim);
        localWeightSim = nullptr;
    }
}
/* *************************************************************** */
void F3dContent::AllocateWarpedGradient() {
    warpedGradient = nifti_dup(*deformationField, false);
}
/* *************************************************************** */
void F3dContent::DeallocateWarpedGradient() {
    if (warpedGradient) {
        nifti_image_free(warpedGradient);
        warpedGradient = nullptr;
    }
}
/* *************************************************************** */
void F3dContent::AllocateTransformationGradient() {
    transformationGradient = nifti_dup(*controlPointGrid, false);
}
/* *************************************************************** */
void F3dContent::DeallocateTransformationGradient() {
    if (transformationGradient != nullptr) {
        nifti_image_free(transformationGradient);
        transformationGradient = nullptr;
    }
}
/* *************************************************************** */
void F3dContent::AllocateVoxelBasedMeasureGradient() {
    voxelBasedMeasureGradient = nifti_dup(*deformationField, false);
}
/* *************************************************************** */
void F3dContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradient) {
        nifti_image_free(voxelBasedMeasureGradient);
        voxelBasedMeasureGradient = nullptr;
    }
}
/* *************************************************************** */
void F3dContent::ZeroTransformationGradient() {
    memset(transformationGradient->data, 0, transformationGradient->nvox * transformationGradient->nbyper);
}
/* *************************************************************** */
void F3dContent::ZeroVoxelBasedMeasureGradient() {
    memset(voxelBasedMeasureGradient->data, 0, voxelBasedMeasureGradient->nvox * voxelBasedMeasureGradient->nbyper);
}
/* *************************************************************** */
