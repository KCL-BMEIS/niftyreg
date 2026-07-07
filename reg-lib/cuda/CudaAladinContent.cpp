#include "CudaAladinContent.h"

/* *************************************************************** */
CudaAladinContent::CudaAladinContent(NiftiImage& referenceIn,
                                     NiftiImage& floatingIn,
                                     int *referenceMaskIn,
                                     mat44 *transformationMatrixIn,
                                     size_t bytesIn,
                                     const unsigned percentageOfBlocks,
                                     const unsigned inlierLts,
                                     int blockStepSize) :
    AladinContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float),
                  percentageOfBlocks, inlierLts, blockStepSize),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    if (bytesIn != sizeof(float))
        NR_WARN_WFCT("Datatype has been forced to float");
    AllocateMask();
    AllocateReferenceMat();
    AllocateBlockMatchingParams();
}
/* *************************************************************** */
CudaAladinContent::~CudaAladinContent() {
    DeallocateMask();
    DeallocateReferenceMat();
    DeallocateBlockMatchingParams();
}
/* *************************************************************** */
void CudaAladinContent::AllocateMask() {
    if (!referenceMask) return;
    Cuda::Allocate(&maskCuda, reference->nvox);
    Cuda::TransferNiftiToDevice(maskCuda, referenceMask, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::DeallocateMask() {
    if (maskCuda) {
        Cuda::Free(maskCuda);
        maskCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaAladinContent::AllocateReferenceMat() {
    float referenceMatCptr[sizeof(mat44) / sizeof(float)];
    mat44ToCptr(*GetXYZMatrix(*reference), referenceMatCptr);
    Cuda::Allocate(&referenceMatCuda, sizeof(mat44) / sizeof(float));
    Cuda::TransferFromHostToDevice(referenceMatCuda, referenceMatCptr, sizeof(mat44) / sizeof(float));
}
/* *************************************************************** */
void CudaAladinContent::DeallocateReferenceMat() {
    if (referenceMatCuda) {
        Cuda::Free(referenceMatCuda);
        referenceMatCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaAladinContent::AllocateBlockMatchingParams() {
    if (!blockMatchingParams) return;
    const size_t positionSize = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
    if (blockMatchingParams->referencePosition) {
        Cuda::Allocate(&referencePositionCuda, positionSize);
        Cuda::TransferFromHostToDevice(referencePositionCuda, blockMatchingParams->referencePosition, positionSize);
    }
    if (blockMatchingParams->warpedPosition) {
        Cuda::Allocate(&warpedPositionCuda, positionSize);
        Cuda::TransferFromHostToDevice(warpedPositionCuda, blockMatchingParams->warpedPosition, positionSize);
    }
    if (blockMatchingParams->totalBlock) {
        Cuda::Allocate(&totalBlockCuda, blockMatchingParams->totalBlockNumber);
        Cuda::TransferFromHostToDevice(totalBlockCuda, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
void CudaAladinContent::DeallocateBlockMatchingParams() {
    if (referencePositionCuda) {
        Cuda::Free(referencePositionCuda);
        referencePositionCuda = nullptr;
    }
    if (warpedPositionCuda) {
        Cuda::Free(warpedPositionCuda);
        warpedPositionCuda = nullptr;
    }
    if (totalBlockCuda) {
        Cuda::Free(totalBlockCuda);
        totalBlockCuda = nullptr;
    }
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    const size_t positionSize = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
    Cuda::TransferFromDeviceToHost(blockMatchingParams->warpedPosition, warpedPositionCuda, positionSize);
    Cuda::TransferFromDeviceToHost(blockMatchingParams->referencePosition, referencePositionCuda, positionSize);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    // Maintain both representations: CudaContent's compacted active-voxel list and the dense mask
    CudaContent::SetReferenceMask(referenceMaskIn);
    DeallocateMask();
    AllocateMask();
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    DeallocateBlockMatchingParams();
    AllocateBlockMatchingParams();
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
