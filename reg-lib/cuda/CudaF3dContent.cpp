#include "CudaF3dContent.h"

/* *************************************************************** */
CudaF3dContent::CudaF3dContent(nifti_image *referenceIn,
                               nifti_image *floatingIn,
                               nifti_image *controlPointGridIn,
                               nifti_image *localWeightSimIn,
                               int *referenceMaskIn,
                               mat44 *transformationMatrixIn,
                               size_t bytesIn):
    F3dContent(referenceIn, floatingIn, controlPointGridIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateControlPointGrid();
    AllocateWarpedGradient();
    AllocateTransformationGradient();
    AllocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
CudaF3dContent::~CudaF3dContent() {
    GetControlPointGrid();  // Transfer device data back to nifti
    DeallocateControlPointGrid();
    DeallocateWarpedGradient();
    DeallocateTransformationGradient();
    DeallocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
void CudaF3dContent::AllocateControlPointGrid() {
    cudaCommon_allocateArrayToDevice(&controlPointGridCuda, controlPointGrid->dim);
    cudaCommon_transferNiftiToArrayOnDevice(controlPointGridCuda, controlPointGrid);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateControlPointGrid() {
    if (controlPointGridCuda) {
        cudaCommon_free(controlPointGridCuda);
        controlPointGridCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateWarpedGradient() {
    cudaCommon_allocateArrayToDevice(&warpedGradientCuda, warpedGradient->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateWarpedGradient() {
    if (warpedGradientCuda != nullptr) {
        cudaCommon_free(warpedGradientCuda);
        warpedGradientCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateTransformationGradient() {
    cudaCommon_allocateArrayToDevice(&transformationGradientCuda, transformationGradient->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateTransformationGradient() {
    if (transformationGradientCuda) {
        cudaCommon_free(transformationGradientCuda);
        transformationGradientCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateVoxelBasedMeasureGradient() {
    cudaCommon_allocateArrayToDevice(&voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradientCuda) {
        cudaCommon_free(voxelBasedMeasureGradientCuda);
        voxelBasedMeasureGradientCuda = nullptr;
    }
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetControlPointGrid() {
    cudaCommon_transferFromDeviceToNifti(controlPointGrid, controlPointGridCuda);
    return controlPointGrid;
}
/* *************************************************************** */
void CudaF3dContent::UpdateControlPointGrid() {
    cudaCommon_transferNiftiToArrayOnDevice(controlPointGridCuda, controlPointGrid);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetTransformationGradient() {
    cudaCommon_transferFromDeviceToNifti(transformationGradient, transformationGradientCuda);
    return transformationGradient;
}
/* *************************************************************** */
void CudaF3dContent::UpdateTransformationGradient() {
    cudaCommon_transferNiftiToArrayOnDevice(transformationGradientCuda, transformationGradient);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetVoxelBasedMeasureGradient() {
    cudaCommon_transferFromDeviceToNifti(voxelBasedMeasureGradient, voxelBasedMeasureGradientCuda);
    return voxelBasedMeasureGradient;
}
/* *************************************************************** */
void CudaF3dContent::UpdateVoxelBasedMeasureGradient() {
    cudaCommon_transferNiftiToArrayOnDevice(voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetWarpedGradient() {
    cudaCommon_transferFromDeviceToNifti(warpedGradient, warpedGradientCuda);
    return warpedGradient;
}
/* *************************************************************** */
void CudaF3dContent::UpdateWarpedGradient() {
    cudaCommon_transferNiftiToArrayOnDevice(warpedGradientCuda, warpedGradient);
}
/* *************************************************************** */
void CudaF3dContent::ZeroTransformationGradient() {
    cudaMemset(transformationGradientCuda, 0,
               transformationGradient->nx * transformationGradient->ny * transformationGradient->nz *
               sizeof(float4));
}
/* *************************************************************** */
void CudaF3dContent::ZeroVoxelBasedMeasureGradient() {
    cudaMemset(voxelBasedMeasureGradientCuda, 0,
               voxelBasedMeasureGradient->nx * voxelBasedMeasureGradient->ny * voxelBasedMeasureGradient->nz *
               sizeof(float4));
}
/* *************************************************************** */
