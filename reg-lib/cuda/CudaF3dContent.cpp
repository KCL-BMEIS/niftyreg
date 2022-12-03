#include "CudaF3dContent.h"

/* *************************************************************** */
CudaF3dContent::CudaF3dContent(nifti_image *referenceIn,
                               nifti_image *floatingIn,
                               nifti_image *controlPointGridIn,
                               nifti_image *localWeightSimIn,
                               int *referenceMaskIn,
                               mat44 *transformationMatrixIn,
                               size_t bytesIn) :
    F3dContent(referenceIn, floatingIn, controlPointGridIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    SetControlPointGrid(controlPointGrid);
    AllocateWarpedGradient();
    AllocateTransformationGradient();
    AllocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
CudaF3dContent::~CudaF3dContent() {
    SetControlPointGrid(nullptr);
    DeallocateWarpedGradient();
    DeallocateTransformationGradient();
    DeallocateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
void CudaF3dContent::AllocateWarpedGradient() {
    if (floating->nt >= 1)
        NR_CUDA_SAFE_CALL(cudaMalloc(&warpedGradientCuda[0], warpedGradient->nvox * sizeof(float4)));
    if (floating->nt == 2)
        NR_CUDA_SAFE_CALL(cudaMalloc(&warpedGradientCuda[1], warpedGradient->nvox * sizeof(float4)));
}
/* *************************************************************** */
void CudaF3dContent::DeallocateWarpedGradient() {
    if (warpedGradientCuda[0] != nullptr) {
        cudaCommon_free(&warpedGradientCuda[0]);
        warpedGradientCuda[0] = nullptr;
    }
    if (warpedGradientCuda[1] != nullptr) {
        cudaCommon_free(&warpedGradientCuda[1]);
        warpedGradientCuda[1] = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateTransformationGradient() {
    cudaCommon_allocateArrayToDevice(&transformationGradientCuda, controlPointGrid->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateTransformationGradient() {
    if (transformationGradientCuda) {
        cudaCommon_free(&transformationGradientCuda);
        transformationGradientCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateVoxelBasedMeasureGradient() {
    cudaCommon_allocateArrayToDevice(&voxelBasedMeasureGradientCuda, reference->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradientCuda) {
        cudaCommon_free(&voxelBasedMeasureGradientCuda);
        voxelBasedMeasureGradientCuda = nullptr;
    }
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetControlPointGrid() {
    cudaCommon_transferFromDeviceToNifti(controlPointGrid, &controlPointGridCuda);
    return controlPointGrid;
}
/* *************************************************************** */
void CudaF3dContent::SetControlPointGrid(nifti_image *controlPointGridIn) {
    F3dContent::SetControlPointGrid(controlPointGridIn);

    if (controlPointGridCuda) {
        cudaCommon_free(&controlPointGridCuda);
        controlPointGridCuda = nullptr;
    }

    if (!controlPointGrid) return;

    cudaCommon_allocateArrayToDevice(&controlPointGridCuda, controlPointGrid->dim);
    cudaCommon_transferNiftiToArrayOnDevice(&controlPointGridCuda, controlPointGrid);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetTransformationGradient() {
    cudaCommon_transferFromDeviceToNifti(transformationGradient, &transformationGradientCuda);
    return transformationGradient;
}
/* *************************************************************** */
void CudaF3dContent::SetTransformationGradient(nifti_image *transformationGradientIn) {
    F3dContent::SetTransformationGradient(transformationGradientIn);
    DeallocateTransformationGradient();
    if (!transformationGradient) return;

    AllocateTransformationGradient();
    cudaCommon_transferNiftiToArrayOnDevice(&transformationGradientCuda, transformationGradient);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetVoxelBasedMeasureGradient() {
    cudaCommon_transferFromDeviceToNifti(voxelBasedMeasureGradient, &voxelBasedMeasureGradientCuda);
    return voxelBasedMeasureGradient;
}
/* *************************************************************** */
void CudaF3dContent::SetVoxelBasedMeasureGradient(nifti_image *voxelBasedMeasureGradientIn) {
    F3dContent::SetVoxelBasedMeasureGradient(voxelBasedMeasureGradientIn);
    DeallocateVoxelBasedMeasureGradient();
    if (!voxelBasedMeasureGradient) return;

    AllocateVoxelBasedMeasureGradient();
    cudaCommon_transferNiftiToArrayOnDevice(&voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient);
}
/* *************************************************************** */
nifti_image* CudaF3dContent::GetWarpedGradient() {
    cudaCommon_transferFromDeviceToNifti(warpedGradient, &warpedGradientCuda[0]);
    return warpedGradient;
}
/* *************************************************************** */
void CudaF3dContent::SetWarpedGradient(nifti_image *warpedGradientIn) {
    F3dContent::SetWarpedGradient(warpedGradientIn);
    DeallocateWarpedGradient();
    if (!warpedGradient) return;

    AllocateWarpedGradient();
    cudaCommon_transferNiftiToArrayOnDevice(&warpedGradientCuda[0], warpedGradient);
    if (warpedGradientCuda[1])
        cudaCommon_transferNiftiToArrayOnDevice(&warpedGradientCuda[1], warpedGradient);
}
/* *************************************************************** */
void CudaF3dContent::ZeroTransformationGradient() {
    cudaMemset(transformationGradientCuda, 0, transformationGradient->nvox * sizeof(float4));
}
/* *************************************************************** */
void CudaF3dContent::ZeroVoxelBasedMeasureGradient() {
    cudaMemset(voxelBasedMeasureGradientCuda, 0, voxelBasedMeasureGradient->nvox * sizeof(float4));
}
/* *************************************************************** */
