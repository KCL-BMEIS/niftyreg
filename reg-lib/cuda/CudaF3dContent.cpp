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
    CudaDefContent(referenceIn, floatingIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    DefContent(referenceIn, floatingIn, localWeightSimIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateControlPointGrid();
    AllocateTransformationGradient();
}
/* *************************************************************** */
CudaF3dContent::~CudaF3dContent() {
    GetControlPointGrid();  // Transfer device data back to nifti
    DeallocateControlPointGrid();
    DeallocateTransformationGradient();
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
void CudaF3dContent::ZeroTransformationGradient() {
    cudaMemset(transformationGradientCuda, 0, NiftiImage::calcVoxelNumber(transformationGradient, 3) * sizeof(float4));
}
/* *************************************************************** */
