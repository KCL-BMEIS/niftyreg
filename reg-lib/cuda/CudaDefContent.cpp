#include "CudaDefContent.h"

/* *************************************************************** */
CudaDefContent::CudaDefContent(nifti_image *referenceIn,
                               nifti_image *floatingIn,
                               nifti_image *localWeightSimIn,
                               int *referenceMaskIn,
                               mat44 *transformationMatrixIn,
                               size_t bytesIn):
    DefContent(referenceIn, floatingIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateWarpedGradient();
    AllocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
CudaDefContent::~CudaDefContent() {
    DeallocateWarpedGradient();
    DeallocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
void CudaDefContent::AllocateWarpedGradient() {
    cudaCommon_allocateArrayToDevice(&warpedGradientCuda, warpedGradient->dim);
}
/* *************************************************************** */
void CudaDefContent::DeallocateWarpedGradient() {
    if (warpedGradientCuda != nullptr) {
        cudaCommon_free(warpedGradientCuda);
        warpedGradientCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaDefContent::AllocateVoxelBasedMeasureGradient() {
    cudaCommon_allocateArrayToDevice(&voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient->dim);
}
/* *************************************************************** */
void CudaDefContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradientCuda) {
        cudaCommon_free(voxelBasedMeasureGradientCuda);
        voxelBasedMeasureGradientCuda = nullptr;
    }
}
/* *************************************************************** */
nifti_image* CudaDefContent::GetVoxelBasedMeasureGradient() {
    cudaCommon_transferFromDeviceToNifti(voxelBasedMeasureGradient, voxelBasedMeasureGradientCuda);
    return voxelBasedMeasureGradient;
}
/* *************************************************************** */
void CudaDefContent::UpdateVoxelBasedMeasureGradient() {
    cudaCommon_transferNiftiToArrayOnDevice(voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient);
}
/* *************************************************************** */
nifti_image* CudaDefContent::GetWarpedGradient() {
    cudaCommon_transferFromDeviceToNifti(warpedGradient, warpedGradientCuda);
    return warpedGradient;
}
/* *************************************************************** */
void CudaDefContent::UpdateWarpedGradient() {
    cudaCommon_transferNiftiToArrayOnDevice(warpedGradientCuda, warpedGradient);
}
/* *************************************************************** */
void CudaDefContent::ZeroVoxelBasedMeasureGradient() {
    cudaMemset(voxelBasedMeasureGradientCuda, 0, NiftiImage::calcVoxelNumber(voxelBasedMeasureGradient, 3) * sizeof(float4));
}
/* *************************************************************** */
