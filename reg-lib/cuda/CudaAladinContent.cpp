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
void CudaAladinContent::AllocateMask() {
    if (!referenceMask) return;
    int *mask;
    Cuda::Allocate(&mask, reference->nvox);
    maskCuda.reset(mask);
    Cuda::TransferNiftiToDevice(maskCuda.get(), referenceMask, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::AllocateReferenceMat() {
    float referenceMatCptr[sizeof(mat44) / sizeof(float)];
    mat44ToCptr(*GetXYZMatrix(*reference), referenceMatCptr);
    float *referenceMat;
    Cuda::Allocate(&referenceMat, sizeof(mat44) / sizeof(float));
    referenceMatCuda.reset(referenceMat);
    Cuda::TransferFromHostToDevice(referenceMatCuda.get(), referenceMatCptr, sizeof(mat44) / sizeof(float));
}
/* *************************************************************** */
void CudaAladinContent::AllocateBlockMatchingParams() {
    if (!blockMatchingParams) return;
    const size_t positionSize = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
    if (blockMatchingParams->referencePosition) {
        float *referencePosition;
        Cuda::Allocate(&referencePosition, positionSize);
        referencePositionCuda.reset(referencePosition);
        Cuda::TransferFromHostToDevice(referencePositionCuda.get(), blockMatchingParams->referencePosition, positionSize);
    }
    if (blockMatchingParams->warpedPosition) {
        float *warpedPosition;
        Cuda::Allocate(&warpedPosition, positionSize);
        warpedPositionCuda.reset(warpedPosition);
        Cuda::TransferFromHostToDevice(warpedPositionCuda.get(), blockMatchingParams->warpedPosition, positionSize);
    }
    if (blockMatchingParams->totalBlock) {
        int *totalBlock;
        Cuda::Allocate(&totalBlock, blockMatchingParams->totalBlockNumber);
        totalBlockCuda.reset(totalBlock);
        Cuda::TransferFromHostToDevice(totalBlockCuda.get(), blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    const size_t positionSize = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
    Cuda::TransferFromDeviceToHost(blockMatchingParams->warpedPosition, warpedPositionCuda.get(), positionSize);
    Cuda::TransferFromDeviceToHost(blockMatchingParams->referencePosition, referencePositionCuda.get(), positionSize);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    // Maintain both representations: CudaContent's compacted active-voxel list and the dense mask
    CudaContent::SetReferenceMask(referenceMaskIn);
    maskCuda = nullptr;
    AllocateMask();
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    referencePositionCuda = nullptr;
    warpedPositionCuda = nullptr;
    totalBlockCuda = nullptr;
    AllocateBlockMatchingParams();
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
