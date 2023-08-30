#include "CudaAladinContent.h"
#include "CudaCommon.hpp"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *referenceIn,
                                     nifti_image *floatingIn,
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

    if (reference != nullptr && reference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(reference);
    if (floating != nullptr && floating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(floating);
        if (warped != nullptr)
            reg_tools_changeDatatype<float>(warped);
    }

    //numBlocks = (blockMatchingParams->activeBlock != nullptr) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void CudaAladinContent::AllocateCuPtrs() {
    if (transformationMatrix != nullptr) {
        Cuda::Allocate<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));

        float *tmpMat_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*(transformationMatrix), tmpMat_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));

        free(tmpMat_h);
    }
    if (referenceMask != nullptr) {
        Cuda::Allocate<int>(&mask_d, reference->nvox);
        Cuda::TransferNiftiToDevice(mask_d, referenceMask, reference->nvox);
    }
    if (reference != nullptr) {
        Cuda::Allocate<float>(&referenceImageArray_d, reference->nvox);
        Cuda::Allocate<float>(&referenceMat_d, sizeof(mat44) / sizeof(float));

        Cuda::TransferNiftiToDevice(referenceImageArray_d, reference);

        float* targetMat = (float *)malloc(sizeof(mat44)); //freed
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        Cuda::TransferNiftiToDevice(referenceMat_d, targetMat, sizeof(mat44) / sizeof(float));
        free(targetMat);
    }
    if (warped != nullptr) {
        Cuda::Allocate<float>(&warpedImageArray_d, warped->nvox);
        Cuda::TransferNiftiToDevice(warpedImageArray_d, warped);
    }
    if (deformationField != nullptr) {
        Cuda::Allocate<float>(&deformationFieldArray_d, deformationField->nvox);
        Cuda::TransferNiftiToDevice(deformationFieldArray_d, deformationField);
    }
    if (floating != nullptr) {
        Cuda::Allocate<float>(&floatingImageArray_d, floating->nvox);
        Cuda::Allocate<float>(&floIJKMat_d, sizeof(mat44) / sizeof(float));

        Cuda::TransferNiftiToDevice(floatingImageArray_d, floating);

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(*floating), sourceIJKMatrix_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, sizeof(mat44), cudaMemcpyHostToDevice));
        free(sourceIJKMatrix_h);
    }

    if (blockMatchingParams != nullptr) {
        if (blockMatchingParams->referencePosition != nullptr) {
            Cuda::Allocate<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->warpedPosition != nullptr) {
            Cuda::Allocate<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            Cuda::TransferFromHostToDevice<float>(warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->totalBlock != nullptr) {
            Cuda::Allocate<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
            Cuda::TransferNiftiToDevice(totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
        /* // Removed until CUDA SVD is added back
        if (blockMatchingParams->activeBlockNumber > 0 ) {
           unsigned m = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
           unsigned n = 0;

           if (blockMatchingParams->dim == 2) {
              n = 6;
           }
           else {
              n = 12;
           }

           Cuda::Allocate<float>(&AR_d, m * n);
           Cuda::Allocate<float>(&U_d, m * m); //only the singular vectors output is needed
           Cuda::Allocate<float>(&VT_d, n * n);
           Cuda::Allocate<float>(&Sigma_d, std::min(m, n));
           Cuda::Allocate<float>(&lengths_d, blockMatchingParams->activeBlockNumber);
           Cuda::Allocate<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        */
    }
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetWarped() {
    DownloadImage(warped, warpedImageArray_d, warped->datatype);
    return warped;
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetDeformationField() {
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
    if (transformationMatrix != nullptr)
        Cuda::Free(transformationMatrix_d);

    AladinContent::SetTransformationMatrix(transformationMatrixIn);
    float *tmpMat_h = (float*)malloc(sizeof(mat44));
    mat44ToCptr(*transformationMatrix, tmpMat_h);

    Cuda::Allocate<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));
    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));
    free(tmpMat_h);
}
/* *************************************************************** */
void CudaAladinContent::SetDeformationField(nifti_image *deformationFieldIn) {
    if (deformationField != nullptr)
        Cuda::Free(deformationFieldArray_d);
    AladinContent::SetDeformationField(deformationFieldIn);

    Cuda::Allocate<float>(&deformationFieldArray_d, deformationField->nvox);
    Cuda::TransferNiftiToDevice(deformationFieldArray_d, deformationField);
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    if (referenceMask != nullptr)
        Cuda::Free(mask_d);
    AladinContent::SetReferenceMask(referenceMaskIn);
    Cuda::Allocate<int>(&mask_d, reference->nvox);
    Cuda::TransferNiftiToDevice(mask_d, referenceMaskIn, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetWarped(nifti_image *warped) {
    if (warped != nullptr)
        Cuda::Free(warpedImageArray_d);
    AladinContent::SetWarped(warped);
    reg_tools_changeDatatype<float>(warped);

    Cuda::Allocate<float>(&warpedImageArray_d, warped->nvox);
    Cuda::TransferNiftiToDevice(warpedImageArray_d, warped);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition != nullptr) {
        Cuda::Free(referencePosition_d);
        //referencePosition
        Cuda::Allocate<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->warpedPosition != nullptr) {
        Cuda::Free(warpedPosition_d);
        //warpedPosition
        Cuda::Allocate<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        Cuda::TransferFromHostToDevice<float>(warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->totalBlock != nullptr) {
        Cuda::Free(totalBlock_d);
        //activeBlock
        Cuda::Allocate<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
        Cuda::TransferFromHostToDevice<int>(totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
    /* // Removed until CUDA SVD is added back
     if (blockMatchingParams->activeBlockNumber > 0) {
         unsigned m = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
         unsigned n = 0;

         if (blockMatchingParams->dim == 2) {
             n = 6;
         }
         else {
             n = 12;
         }

         Cuda::Allocate<float>(&AR_d, m * n);
         Cuda::Allocate<float>(&U_d, m * m); //only the singular vectors output is needed
         Cuda::Allocate<float>(&VT_d, n * n);
         Cuda::Allocate<float>(&Sigma_d, std::min(m, n));
         Cuda::Allocate<float>(&lengths_d, blockMatchingParams->activeBlockNumber);
         Cuda::Allocate<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
     }
     */
}
/* *************************************************************** */
template<class DataType>
DataType CudaAladinContent::FillWarpedImageData(float intensity, int datatype) {
    switch (datatype) {
    case NIFTI_TYPE_FLOAT32:
        return static_cast<float>(intensity);
        break;
    case NIFTI_TYPE_FLOAT64:
        return static_cast<double>(intensity);
        break;
    case NIFTI_TYPE_UINT8:
        intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
        return static_cast<unsigned char>(intensity > 0 ? Round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT16:
        intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
        return static_cast<unsigned short>(intensity > 0 ? Round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT32:
        intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
        return static_cast<unsigned>(intensity > 0 ? Round(intensity) : 0);
        break;
    default:
        return static_cast<DataType>(Round(intensity));
        break;
    }
}
/* *************************************************************** */
template<class T>
void CudaAladinContent::FillImageData(nifti_image *image, float *memoryObject, int type) {
    size_t size = image->nvox;
    float *buffer = (float*)malloc(size * sizeof(float));

    Cuda::TransferFromDeviceToHost<float>(buffer, memoryObject, size);

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = malloc(image->nvox * image->nbyper);
    T* dataT = static_cast<T*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = FillWarpedImageData<T>(buffer[i], type);
    free(buffer);
}
/* *************************************************************** */
void CudaAladinContent::DownloadImage(nifti_image *image, float *memoryObject, int datatype) {
    switch (datatype) {
    case NIFTI_TYPE_FLOAT32:
        FillImageData<float>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_FLOAT64:
        FillImageData<double>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT8:
        FillImageData<unsigned char>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT8:
        FillImageData<char>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT16:
        FillImageData<unsigned short>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT16:
        FillImageData<short>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT32:
        FillImageData<unsigned>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT32:
        FillImageData<int>(image, memoryObject, datatype);
        break;
    default:
        NR_FATAL_ERROR("CUDA: unsupported type");
    }
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
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetAR_d()
{
   return AR_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetU_d()
{
   return U_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetVT_d()
{
   return VT_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetSigma_d()
{
   return Sigma_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetLengths_d()
{
   return lengths_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::GetNewWarpedPos_d()
{
   return newWarpedPos_d;
}
*/
/* *************************************************************** */
int* CudaAladinContent::GetTotalBlock_d() {
    return totalBlock_d;
}
/* *************************************************************** */
int* CudaAladinContent::GetMask_d() {
    return mask_d;
}
/* *************************************************************** */
int* CudaAladinContent::GetReferenceDims() {
    return referenceDims;
}
/* *************************************************************** */
int* CudaAladinContent::GetFloatingDims() {
    return floatingDims;
}
/* *************************************************************** */
void CudaAladinContent::FreeCuPtrs() {
    if (transformationMatrix_d != nullptr)
        Cuda::Free(transformationMatrix_d);

    if (referenceImageArray_d != nullptr)
        Cuda::Free(referenceImageArray_d);
    if (referenceMat_d != nullptr)
        Cuda::Free(referenceMat_d);

    if (floatingImageArray_d != nullptr)
        Cuda::Free(floatingImageArray_d);
    if (floIJKMat_d != nullptr)
        Cuda::Free(floIJKMat_d);

    if (warpedImageArray_d != nullptr)
        Cuda::Free(warpedImageArray_d);

    if (deformationFieldArray_d != nullptr)
        Cuda::Free(deformationFieldArray_d);

    if (mask_d != nullptr)
        Cuda::Free(mask_d);

    if (totalBlock_d != nullptr)
        Cuda::Free(totalBlock_d);
    if (referencePosition_d != nullptr)
        Cuda::Free(referencePosition_d);
    if (warpedPosition_d != nullptr)
        Cuda::Free(warpedPosition_d);
        /*
        Cuda::Free(AR_d);
        Cuda::Free(U_d);
        Cuda::Free(VT_d);
        Cuda::Free(Sigma_d);
        Cuda::Free(lengths_d);
        Cuda::Free(newWarpedPos_d);
        */
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
