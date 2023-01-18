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
    AllocateLocalWeightSim(localWeightSimIn);
    AllocateWarpedGradient();
    AllocateTransformationGradient();
    AllocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
F3dContent::~F3dContent() {
    DeallocateLocalWeightSim();
    DeallocateWarpedGradient();
    DeallocateTransformationGradient();
    DeallocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
void F3dContent::AllocateLocalWeightSim(nifti_image *localWeightSimIn) {
    if (!localWeightSimIn) return;
    localWeightSim = nifti_copy_nim_info(reference);
    localWeightSim->dim[0] = localWeightSim->ndim = localWeightSimIn->dim[0];
    localWeightSim->dim[4] = localWeightSim->nt = localWeightSimIn->dim[4];
    localWeightSim->dim[5] = localWeightSim->nu = localWeightSimIn->dim[5];
    localWeightSim->nvox = size_t(localWeightSim->nx * localWeightSim->ny * localWeightSim->nz *
                                  localWeightSim->nt * localWeightSim->nu);
    localWeightSim->data = malloc(localWeightSim->nvox * localWeightSim->nbyper);
    F3dContent::ZeroVoxelBasedMeasureGradient();
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
    warpedGradient = nifti_copy_nim_info(deformationField);
    warpedGradient->data = calloc(warpedGradient->nvox, warpedGradient->nbyper);
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
    transformationGradient = nifti_copy_nim_info(controlPointGrid);
    transformationGradient->data = calloc(transformationGradient->nvox, transformationGradient->nbyper);
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
    voxelBasedMeasureGradient = nifti_copy_nim_info(deformationField);
    voxelBasedMeasureGradient->data = calloc(voxelBasedMeasureGradient->nvox, voxelBasedMeasureGradient->nbyper);
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
