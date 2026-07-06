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
    // The virtual Content base is constructed by this most-derived class; CudaContent owns the shared
    // device storage (images, float4 deformation, compacted mask, transformation) and its up/downloads.
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
    referencePosition_d = nullptr;
    warpedPosition_d = nullptr;
    totalBlock_d = nullptr;
    mask_d = nullptr;
    referenceMat_d = nullptr;
}
/* *************************************************************** */
void CudaAladinContent::AllocateCuPtrs() {
    // Dense per-voxel mask for the affine / block-matching kernels (CudaContent holds the compacted
    // active-voxel list used by the resampler).
    if (referenceMask) {
        Cuda::Allocate<int>(&mask_d, reference->nvox);
        Cuda::TransferNiftiToDevice(mask_d, referenceMask, reference->nvox);
    }
    // Reference XYZ matrix used by block matching.
    if (reference) {
        Cuda::Allocate<float>(&referenceMat_d, sizeof(mat44) / sizeof(float));
        float* targetMat = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        Cuda::TransferNiftiToDevice(referenceMat_d, targetMat, sizeof(mat44) / sizeof(float));
        free(targetMat);
    }
    if (blockMatchingParams) {
        if (blockMatchingParams->referencePosition) {
            Cuda::Allocate<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->warpedPosition) {
            Cuda::Allocate<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->totalBlock) {
            Cuda::Allocate<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
            Cuda::TransferNiftiToDevice(totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
    }
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->warpedPosition, warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->referencePosition, referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    // Maintain both representations: CudaContent's compacted active-voxel list (used by the resampler)
    // and the dense per-voxel mask (used by the affine and block-matching kernels).
    CudaContent::SetReferenceMask(referenceMaskIn);
    if (mask_d) {
        Cuda::Free(mask_d);
        mask_d = nullptr;
    }
    if (!referenceMask) return;
    Cuda::Allocate<int>(&mask_d, reference->nvox);
    Cuda::TransferNiftiToDevice(mask_d, referenceMask, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition) {
        Cuda::Free(referencePosition_d);
        Cuda::Allocate<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->warpedPosition) {
        Cuda::Free(warpedPosition_d);
        Cuda::Allocate<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->totalBlock) {
        Cuda::Free(totalBlock_d);
        Cuda::Allocate<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
        Cuda::TransferFromHostToDevice<int>(totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
void CudaAladinContent::FreeCuPtrs() {
    if (referenceMat_d)
        Cuda::Free(referenceMat_d);
    if (mask_d)
        Cuda::Free(mask_d);
    if (totalBlock_d)
        Cuda::Free(totalBlock_d);
    if (referencePosition_d)
        Cuda::Free(referencePosition_d);
    if (warpedPosition_d)
        Cuda::Free(warpedPosition_d);
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
