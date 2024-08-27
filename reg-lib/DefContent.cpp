#include "DefContent.h"
#include "_reg_resampling.h"

/* *************************************************************** */
DefContent::DefContent(NiftiImage& referenceIn,
                       NiftiImage& floatingIn,
                       NiftiImage *localWeightSimIn,
                       int *referenceMaskIn,
                       mat44 *transformationMatrixIn,
                       size_t bytesIn):
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn) {
    warpedGradient = NiftiImage(deformationField, NiftiImage::Copy::ImageInfoAndAllocData);
    voxelBasedMeasureGradient = NiftiImage(deformationField, NiftiImage::Copy::ImageInfoAndAllocData);
    if (localWeightSimIn && *localWeightSimIn)
        AllocateLocalWeightSim(*localWeightSimIn);
}
/* *************************************************************** */
void DefContent::AllocateLocalWeightSim(NiftiImage& localWeightSimIn) {
    localWeightSim = NiftiImage(reference, NiftiImage::Copy::ImageInfo);
    localWeightSim.setDim(NiftiDim::NDim, localWeightSimIn->dim[0]);
    localWeightSim.setDim(NiftiDim::T, localWeightSimIn->dim[4]);
    localWeightSim.setDim(NiftiDim::U, localWeightSimIn->dim[5]);
    localWeightSim.realloc();
    reg_getDeformationFromDisplacement(voxelBasedMeasureGradient);
    reg_resampleImage(localWeightSimIn, localWeightSim, voxelBasedMeasureGradient, nullptr, 1, 0);
}
/* *************************************************************** */
void DefContent::ZeroVoxelBasedMeasureGradient() {
    memset(voxelBasedMeasureGradient->data, 0, voxelBasedMeasureGradient.totalBytes());
}
/* *************************************************************** */
