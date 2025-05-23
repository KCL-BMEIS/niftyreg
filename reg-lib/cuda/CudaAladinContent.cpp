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
    AladinContent(referenceIn,
                  floatingIn,
                  referenceMaskIn,
                  transformationMatrixIn,
                  sizeof(float), // forcing float for CUDA
                  percentageOfBlocks,
                  inlierLts,
                  blockStepSize) {
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
    referenceImageArray_d = nullptr;
    floatingImageArray_d = nullptr;
    transformationMatrix_d = nullptr;
    warpedImageArray_d = nullptr;
    deformationFieldArray_d = nullptr;
    referencePosition_d = nullptr;
    warpedPosition_d = nullptr;
    totalBlock_d = nullptr;
    mask_d = nullptr;
    floIJKMat_d = nullptr;

    if (reference && reference->nbyper != NIFTI_TYPE_FLOAT32)
        reference.changeDatatype(NIFTI_TYPE_FLOAT32);
    if (floating && floating->nbyper != NIFTI_TYPE_FLOAT32) {
        floating.changeDatatype(NIFTI_TYPE_FLOAT32);
        if (warped)
            warped.changeDatatype(NIFTI_TYPE_FLOAT32);
    }
}
/* *************************************************************** */
void CudaAladinContent::AllocateCuPtrs() {
    if (transformationMatrix) {
        Cuda::Allocate<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));

        float *tmpMat_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*(transformationMatrix), tmpMat_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));

        free(tmpMat_h);
    }
    if (referenceMask) {
        Cuda::Allocate<int>(&mask_d, reference->nvox);
        Cuda::TransferNiftiToDevice(mask_d, referenceMask, reference->nvox);
    }
    if (reference) {
        Cuda::Allocate<float>(&referenceImageArray_d, reference->nvox);
        Cuda::Allocate<float>(&referenceMat_d, sizeof(mat44) / sizeof(float));

        Cuda::TransferNiftiToDevice(referenceImageArray_d, reference);

        float* targetMat = (float *)malloc(sizeof(mat44)); //freed
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        Cuda::TransferNiftiToDevice(referenceMat_d, targetMat, sizeof(mat44) / sizeof(float));
        free(targetMat);
    }
    if (warped) {
        Cuda::Allocate<float>(&warpedImageArray_d, warped->nvox);
        Cuda::TransferNiftiToDevice(warpedImageArray_d, warped);
    }
    if (deformationField) {
        Cuda::Allocate<float>(&deformationFieldArray_d, deformationField->nvox);
        Cuda::TransferNiftiToDevice(deformationFieldArray_d, deformationField);
    }
    if (floating) {
        Cuda::Allocate<float>(&floatingImageArray_d, floating->nvox);
        Cuda::Allocate<float>(&floIJKMat_d, sizeof(mat44) / sizeof(float));

        Cuda::TransferNiftiToDevice(floatingImageArray_d, floating);

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(*floating), sourceIJKMatrix_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, sizeof(mat44), cudaMemcpyHostToDevice));
        free(sourceIJKMatrix_h);
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
NiftiImage& CudaAladinContent::GetWarped() {
    DownloadImage(warped, warpedImageArray_d, warped->datatype);
    return warped;
}
/* *************************************************************** */
NiftiImage& CudaAladinContent::GetDeformationField() {
    Cuda::TransferFromDeviceToHost<float>((float*)deformationField->data, deformationFieldArray_d, deformationField->nvox);
    return deformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->warpedPosition, warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    Cuda::TransferFromDeviceToHost<float>(blockMatchingParams->referencePosition, referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    if (transformationMatrix)
        Cuda::Free(transformationMatrix_d);

    AladinContent::SetTransformationMatrix(transformationMatrixIn);
    float *tmpMat_h = (float*)malloc(sizeof(mat44));
    mat44ToCptr(*transformationMatrix, tmpMat_h);

    Cuda::Allocate<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));
    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));
    free(tmpMat_h);
}
/* *************************************************************** */
void CudaAladinContent::SetDeformationField(NiftiImage&& deformationFieldIn) {
    if (deformationField)
        Cuda::Free(deformationFieldArray_d);
    AladinContent::SetDeformationField(std::move(deformationFieldIn));

    Cuda::Allocate<float>(&deformationFieldArray_d, deformationField->nvox);
    Cuda::TransferNiftiToDevice(deformationFieldArray_d, deformationField);
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    if (referenceMask)
        Cuda::Free(mask_d);
    AladinContent::SetReferenceMask(referenceMaskIn);
    Cuda::Allocate<int>(&mask_d, reference->nvox);
    Cuda::TransferNiftiToDevice(mask_d, referenceMaskIn, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetWarped(NiftiImage&& warpedIn) {
    AladinContent::SetWarped(std::move(warpedIn));
    if (warpedImageArray_d) {
        Cuda::Free(warpedImageArray_d);
        warpedImageArray_d = nullptr;
    }
    if (!warped) return;

    if (warped->nbyper != NIFTI_TYPE_FLOAT32)
        warped.changeDatatype(NIFTI_TYPE_FLOAT32);

    Cuda::Allocate<float>(&warpedImageArray_d, warped->nvox);
    Cuda::TransferNiftiToDevice(warpedImageArray_d, warped);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition) {
        Cuda::Free(referencePosition_d);
        //referencePosition
        Cuda::Allocate<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->warpedPosition) {
        Cuda::Free(warpedPosition_d);
        //warpedPosition
        Cuda::Allocate<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->totalBlock) {
        Cuda::Free(totalBlock_d);
        //activeBlock
        Cuda::Allocate<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
        Cuda::TransferFromHostToDevice<int>(totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
void CudaAladinContent::DownloadImage(NiftiImage& image, float *memoryObject, int datatype) {
    const size_t size = image->nvox;
    unique_ptr<float[]> buffer(new float[size]);

    Cuda::TransferFromDeviceToHost(buffer.get(), memoryObject, size);

    std::visit([&](auto&& dataType) {
        using DataType = std::decay_t<decltype(dataType)>;
        image->datatype = datatype;
        image->nbyper = sizeof(DataType);
        image.realloc();
        DataType *data = static_cast<DataType*>(image->data);
        for (size_t i = 0; i < size; ++i)
            data[i] = image.clampData<DataType>(buffer[i]);
    }, image.getDataType());
}
/* *************************************************************** */
float* CudaAladinContent::GetReferenceImageArray_d() {
    return referenceImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetFloatingImageArray_d() {
    return floatingImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetWarpedImageArray_d() {
    return warpedImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetTransformationMatrix_d() {
    return transformationMatrix_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetReferencePosition_d() {
    return referencePosition_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetWarpedPosition_d() {
    return warpedPosition_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetDeformationFieldArray_d() {
    return deformationFieldArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetReferenceMat_d() {
    return referenceMat_d;
}
/* *************************************************************** */
float* CudaAladinContent::GetFloIJKMat_d() {
    return floIJKMat_d;
}
/* *************************************************************** */
int* CudaAladinContent::GetTotalBlock_d() {
    return totalBlock_d;
}
/* *************************************************************** */
int* CudaAladinContent::GetMask_d() {
    return mask_d;
}
/* *************************************************************** */
void CudaAladinContent::FreeCuPtrs() {
    if (transformationMatrix_d)
        Cuda::Free(transformationMatrix_d);

    if (referenceImageArray_d)
        Cuda::Free(referenceImageArray_d);
    if (referenceMat_d)
        Cuda::Free(referenceMat_d);

    if (floatingImageArray_d)
        Cuda::Free(floatingImageArray_d);
    if (floIJKMat_d)
        Cuda::Free(floIJKMat_d);

    if (warpedImageArray_d)
        Cuda::Free(warpedImageArray_d);

    if (deformationFieldArray_d)
        Cuda::Free(deformationFieldArray_d);

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
