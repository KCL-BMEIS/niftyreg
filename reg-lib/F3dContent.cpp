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
    if (!controlPointGridIn)
        NR_FATAL_ERROR("controlPointGridIn can't be nullptr");
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
