#include "CudaAladinContent.h"
#include "CudaCommon.hpp"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent(NiftiImage& referenceIn,
                                     NiftiImage& floatingIn,
                                     int *referenceMaskIn,
                                     mat44 *transformationMatrixIn,
                                     size_t bytesIn,
                                     const unsigned percentageOfBlocks,
                                     const unsigned inlierLts,
                                     int blockStepSize) :
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)),
    AladinContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float),
                  percentageOfBlocks, inlierLts, blockStepSize),
    CudaContent(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    if (bytesIn != sizeof(float))
        NR_WARN_WFCT("Datatype has been forced to float");
    InitVars();
    AllocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::~CudaAladinContent() {
    FreeCuPtrs();
}
/* *************************************************************** */
void CudaAladinContent::InitVars() {
    referencePositionCuda = nullptr;
    warpedPositionCuda = nullptr;
    totalBlockCuda = nullptr;
    maskCuda = nullptr;
    referenceMatCuda = nullptr;
}
/* *************************************************************** */
void CudaAladinContent::AllocateCuPtrs() {
    // Dense per-voxel mask for the affine / block-matching kernels (CudaContent holds the compacted
    // active-voxel list used by the resampler)
    if (referenceMask) {
        Cuda::Allocate<int>(&maskCuda, reference->nvox);
        Cuda::TransferNiftiToDevice(maskCuda, referenceMask, reference->nvox);
    }
    // Reference XYZ matrix used by block matching
    if (reference) {
        Cuda::Allocate<float>(&referenceMatCuda, sizeof(mat44) / sizeof(float));
        float* targetMat = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        Cuda::TransferNiftiToDevice(referenceMatCuda, targetMat, sizeof(mat44) / sizeof(float));
        free(targetMat);
    }
    if (blockMatchingParams) {
        if (blockMatchingParams->referencePosition) {
            Cuda::Allocate<float>(&referencePositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(referencePositionCuda, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->warpedPosition) {
            Cuda::Allocate<float>(&warpedPositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(warpedPositionCuda, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->totalBlock) {
            Cuda::Allocate<int>(&totalBlockCuda, blockMatchingParams->totalBlockNumber);
            Cuda::TransferNiftiToDevice(totalBlockCuda, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
    }
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->warpedPosition, warpedPositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->referencePosition, referencePositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    // Maintain both representations: CudaContent's compacted active-voxel list and the dense mask
    CudaContent::SetReferenceMask(referenceMaskIn);
    if (maskCuda) {
        Cuda::Free(maskCuda);
        maskCuda = nullptr;
    }
    if (!referenceMask) return;
    Cuda::Allocate<int>(&maskCuda, reference->nvox);
    Cuda::TransferNiftiToDevice(maskCuda, referenceMask, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition) {
        Cuda::Free(referencePositionCuda);
        Cuda::Allocate<float>(&referencePositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(referencePositionCuda, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->warpedPosition) {
        Cuda::Free(warpedPositionCuda);
        Cuda::Allocate<float>(&warpedPositionCuda, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(warpedPositionCuda, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->totalBlock) {
        Cuda::Free(totalBlockCuda);
        Cuda::Allocate<int>(&totalBlockCuda, blockMatchingParams->totalBlockNumber);
        Cuda::TransferFromHostToDevice<int>(totalBlockCuda, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
void CudaAladinContent::FreeCuPtrs() {
    if (referenceMatCuda)
        Cuda::Free(referenceMatCuda);
    if (maskCuda)
        Cuda::Free(maskCuda);
    if (totalBlockCuda)
        Cuda::Free(totalBlockCuda);
    if (referencePositionCuda)
        Cuda::Free(referencePositionCuda);
    if (warpedPositionCuda)
        Cuda::Free(warpedPositionCuda);
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
