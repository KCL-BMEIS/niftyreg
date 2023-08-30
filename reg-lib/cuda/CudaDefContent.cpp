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
    AllocateLocalWeightSim();
}
/* *************************************************************** */
CudaDefContent::~CudaDefContent() {
    DeallocateWarpedGradient();
    DeallocateVoxelBasedMeasureGradient();
    DeallocateLocalWeightSim();
}
/* *************************************************************** */
void CudaDefContent::AllocateLocalWeightSim() {
    if (!localWeightSim) return;
    Cuda::Allocate(&localWeightSimCuda, localWeightSim->nvox);
    Cuda::TransferNiftiToDevice(localWeightSimCuda, localWeightSim);
}
/* *************************************************************** */
void CudaDefContent::DeallocateLocalWeightSim() {
    if (localWeightSimCuda != nullptr) {
        Cuda::Free(localWeightSimCuda);
        localWeightSimCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaDefContent::AllocateWarpedGradient() {
    Cuda::Allocate(&warpedGradientCuda, warpedGradient->dim);
}
/* *************************************************************** */
void CudaDefContent::DeallocateWarpedGradient() {
    if (warpedGradientCuda != nullptr) {
        Cuda::Free(warpedGradientCuda);
        warpedGradientCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaDefContent::AllocateVoxelBasedMeasureGradient() {
    Cuda::Allocate(&voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient->dim);
}
/* *************************************************************** */
void CudaDefContent::DeallocateVoxelBasedMeasureGradient() {
    if (voxelBasedMeasureGradientCuda) {
        Cuda::Free(voxelBasedMeasureGradientCuda);
        voxelBasedMeasureGradientCuda = nullptr;
    }
}
/* *************************************************************** */
nifti_image* CudaDefContent::GetLocalWeightSim() {
    Cuda::TransferFromDeviceToNifti(localWeightSim, localWeightSimCuda);
    return localWeightSim;
}
/* *************************************************************** */
nifti_image* CudaDefContent::GetVoxelBasedMeasureGradient() {
    Cuda::TransferFromDeviceToNifti(voxelBasedMeasureGradient, voxelBasedMeasureGradientCuda);
    return voxelBasedMeasureGradient;
}
/* *************************************************************** */
void CudaDefContent::UpdateVoxelBasedMeasureGradient() {
    Cuda::TransferNiftiToDevice(voxelBasedMeasureGradientCuda, voxelBasedMeasureGradient);
}
/* *************************************************************** */
nifti_image* CudaDefContent::GetWarpedGradient() {
    Cuda::TransferFromDeviceToNifti(warpedGradient, warpedGradientCuda);
    return warpedGradient;
}
/* *************************************************************** */
void CudaDefContent::UpdateWarpedGradient() {
    Cuda::TransferNiftiToDevice(warpedGradientCuda, warpedGradient);
}
/* *************************************************************** */
void CudaDefContent::ZeroVoxelBasedMeasureGradient() {
    cudaMemset(voxelBasedMeasureGradientCuda, 0, NiftiImage::calcVoxelNumber(voxelBasedMeasureGradient, 3) * sizeof(float4));
}
/* *************************************************************** */
