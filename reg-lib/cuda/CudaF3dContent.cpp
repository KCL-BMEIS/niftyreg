#include "CudaF3dContent.h"

/* *************************************************************** */
CudaF3dContent::CudaF3dContent(NiftiImage& referenceIn,
                               NiftiImage& floatingIn,
                               NiftiImage& controlPointGridIn,
                               NiftiImage *localWeightSimIn,
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
    CudaF3dContent::GetControlPointGrid();  // Transfer device data back to nifti
    DeallocateControlPointGrid();
    DeallocateTransformationGradient();
}
/* *************************************************************** */
void CudaF3dContent::AllocateControlPointGrid() {
    Cuda::Allocate(&controlPointGridCuda, controlPointGrid->dim);
    CudaF3dContent::UpdateControlPointGrid();
}
/* *************************************************************** */
void CudaF3dContent::DeallocateControlPointGrid() {
    if (controlPointGridCuda) {
        Cuda::Free(controlPointGridCuda);
        controlPointGridCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaF3dContent::AllocateTransformationGradient() {
    Cuda::Allocate(&transformationGradientCuda, transformationGradient->dim);
}
/* *************************************************************** */
void CudaF3dContent::DeallocateTransformationGradient() {
    if (transformationGradientCuda) {
        Cuda::Free(transformationGradientCuda);
        transformationGradientCuda = nullptr;
    }
}
/* *************************************************************** */
NiftiImage& CudaF3dContent::GetControlPointGrid() {
    Cuda::TransferFromDeviceToNifti(controlPointGrid, controlPointGridCuda);
    return controlPointGrid;
}
/* *************************************************************** */
void CudaF3dContent::UpdateControlPointGrid() {
    Cuda::TransferNiftiToDevice(controlPointGridCuda, controlPointGrid);
}
/* *************************************************************** */
NiftiImage& CudaF3dContent::GetTransformationGradient() {
    Cuda::TransferFromDeviceToNifti(transformationGradient, transformationGradientCuda);
    return transformationGradient;
}
/* *************************************************************** */
void CudaF3dContent::UpdateTransformationGradient() {
    Cuda::TransferNiftiToDevice(transformationGradientCuda, transformationGradient);
}
/* *************************************************************** */
void CudaF3dContent::ZeroTransformationGradient() {
    cudaMemset(transformationGradientCuda, 0, transformationGradient.nVoxelsPerVolume() * sizeof(float4));
}
/* *************************************************************** */
