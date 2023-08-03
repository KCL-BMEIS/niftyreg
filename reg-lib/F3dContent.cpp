#include "F3dContent.h"

/* *************************************************************** */
F3dContent::F3dContent(nifti_image *referenceIn,
                       nifti_image *floatingIn,
                       nifti_image *controlPointGridIn,
                       nifti_image *localWeightSimIn,
                       int *referenceMaskIn,
                       mat44 *transformationMatrixIn,
                       size_t bytesIn):
    DefContent(referenceIn, floatingIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    controlPointGrid(controlPointGridIn) {
    if (!controlPointGridIn) {
        reg_print_fct_error("F3dContent::F3dContent()");
        reg_print_msg_error("controlPointGridIn can't be nullptr");
        reg_exit();
    }
    AllocateTransformationGradient();
}
/* *************************************************************** */
F3dContent::~F3dContent() {
    DeallocateTransformationGradient();
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
void F3dContent::ZeroTransformationGradient() {
    memset(transformationGradient->data, 0, transformationGradient->nvox * transformationGradient->nbyper);
}
/* *************************************************************** */
