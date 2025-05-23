#include "F3dContent.h"

/* *************************************************************** */
F3dContent::F3dContent(NiftiImage& referenceIn,
                       NiftiImage& floatingIn,
                       NiftiImage& controlPointGridIn,
                       NiftiImage *localWeightSimIn,
                       int *referenceMaskIn,
                       mat44 *transformationMatrixIn,
                       size_t bytesIn):
    DefContent(referenceIn, floatingIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    controlPointGrid(NiftiImage(controlPointGridIn, NiftiImage::Copy::Acquire)) {
    if (!controlPointGridIn)
        NR_FATAL_ERROR("controlPointGridIn can't be nullptr");
    transformationGradient = NiftiImage(controlPointGrid, NiftiImage::Copy::ImageInfoAndAllocData);
}
/* *************************************************************** */
void F3dContent::ZeroTransformationGradient() {
    memset(transformationGradient->data, 0, transformationGradient.totalBytes());
}
/* *************************************************************** */
