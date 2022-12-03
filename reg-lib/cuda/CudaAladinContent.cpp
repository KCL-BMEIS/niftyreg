#include "CudaAladinContent.h"
#include "_reg_common_cuda.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *referenceIn,
                                     nifti_image *floatingIn,
                                     int *referenceMaskIn,
                                     mat44 *transformationMatrixIn,
                                     size_t bytesIn,
                                     const unsigned int percentageOfBlocks,
                                     const unsigned int inlierLts,
                                     int blockStepSize) :
    AladinContent(referenceIn,
                  floatingIn,
                  referenceMaskIn,
                  transformationMatrixIn,
                  sizeof(float), // forcing float for CUDA
                  percentageOfBlocks,
                  inlierLts,
                  blockStepSize) {
    if (bytesIn != sizeof(float)) {
        reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
        reg_print_msg_warn("Datatype has been forced to float");
    }
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
        cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));

        float *tmpMat_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*(transformationMatrix), tmpMat_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));

        free(tmpMat_h);
    }
    if (referenceMask != nullptr) {
        cudaCommon_allocateArrayToDevice<int>(&mask_d, reference->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, referenceMask, reference->nvox);
    }
    if (reference != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, reference->nvox);
        cudaCommon_allocateArrayToDevice<float>(&referenceMat_d, sizeof(mat44) / sizeof(float));

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, reference);

        float* targetMat = (float *)malloc(sizeof(mat44)); //freed
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, sizeof(mat44) / sizeof(float));
        free(targetMat);
    }
    if (warped != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, warped->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, warped);
    }
    if (deformationField != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, deformationField->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, deformationField);
    }
    if (floating != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floating->nvox);
        cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, sizeof(mat44) / sizeof(float));

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, floating);

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(*floating), sourceIJKMatrix_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, sizeof(mat44), cudaMemcpyHostToDevice));
        free(sourceIJKMatrix_h);
    }

    if (blockMatchingParams != nullptr) {
        if (blockMatchingParams->referencePosition != nullptr) {
            cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->warpedPosition != nullptr) {
            cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        if (blockMatchingParams->totalBlock != nullptr) {
            cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
            cudaCommon_transferFromDeviceToNiftiSimple1<int>(&totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
        /* // Removed until CUDA SVD is added back
        if (blockMatchingParams->activeBlockNumber > 0 ) {
           unsigned int m = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
           unsigned int n = 0;

           if (blockMatchingParams->dim == 2) {
              n = 6;
           }
           else {
              n = 12;
           }

           cudaCommon_allocateArrayToDevice<float>(&AR_d, m * n);
           cudaCommon_allocateArrayToDevice<float>(&U_d, m * m); //only the singular vectors output is needed
           cudaCommon_allocateArrayToDevice<float>(&VT_d, n * n);
           cudaCommon_allocateArrayToDevice<float>(&Sigma_d, std::min(m, n));
           cudaCommon_allocateArrayToDevice<float>(&lengths_d, blockMatchingParams->activeBlockNumber);
           cudaCommon_allocateArrayToDevice<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        }
        */
    }
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetWarped(int datatype, int index) {
    DownloadImage(warped, warpedImageArray_d, datatype);
    return warped;
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetDeformationField() {
    cudaCommon_transferFromDeviceToCpu<float>((float*)deformationField->data, &deformationFieldArray_d, deformationField->nvox);
    return deformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {
    cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->warpedPosition, &warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->referencePosition, &referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    return blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    if (transformationMatrix != nullptr)
        cudaCommon_free(&transformationMatrix_d);

    AladinContent::SetTransformationMatrix(transformationMatrixIn);
    float *tmpMat_h = (float*)malloc(sizeof(mat44));
    mat44ToCptr(*transformationMatrix, tmpMat_h);

    cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, sizeof(mat44) / sizeof(float));
    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix_d, tmpMat_h, sizeof(mat44), cudaMemcpyHostToDevice));
    free(tmpMat_h);
}
/* *************************************************************** */
void CudaAladinContent::SetDeformationField(nifti_image *deformationFieldIn) {
    if (deformationField != nullptr)
        cudaCommon_free(&deformationFieldArray_d);
    AladinContent::SetDeformationField(deformationFieldIn);

    cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, deformationField->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, deformationField);
}
/* *************************************************************** */
void CudaAladinContent::SetReferenceMask(int *referenceMaskIn) {
    if (referenceMask != nullptr)
        cudaCommon_free(&mask_d);
    AladinContent::SetReferenceMask(referenceMaskIn);
    cudaCommon_allocateArrayToDevice<int>(&mask_d, reference->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, referenceMaskIn, reference->nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetWarped(nifti_image *warped) {
    if (warped != nullptr)
        cudaCommon_free(&warpedImageArray_d);
    AladinContent::SetWarped(warped);
    reg_tools_changeDatatype<float>(warped);

    cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, warped->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, warped);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition != nullptr) {
        cudaCommon_free(&referencePosition_d);
        //referencePosition
        cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, blockMatchingParams->referencePosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->warpedPosition != nullptr) {
        cudaCommon_free(&warpedPosition_d);
        //warpedPosition
        cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, blockMatchingParams->warpedPosition, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
    }
    if (blockMatchingParams->totalBlock != nullptr) {
        cudaCommon_free(&totalBlock_d);
        //activeBlock
        cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
        cudaCommon_transferArrayFromCpuToDevice<int>(&totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
    }
    /* // Removed until CUDA SVD is added back
     if (blockMatchingParams->activeBlockNumber > 0) {
         unsigned int m = blockMatchingParams->activeBlockNumber * blockMatchingParams->dim;
         unsigned int n = 0;

         if (blockMatchingParams->dim == 2) {
             n = 6;
         }
         else {
             n = 12;
         }

         cudaCommon_allocateArrayToDevice<float>(&AR_d, m * n);
         cudaCommon_allocateArrayToDevice<float>(&U_d, m * m); //only the singular vectors output is needed
         cudaCommon_allocateArrayToDevice<float>(&VT_d, n * n);
         cudaCommon_allocateArrayToDevice<float>(&Sigma_d, std::min(m, n));
         cudaCommon_allocateArrayToDevice<float>(&lengths_d, blockMatchingParams->activeBlockNumber);
         cudaCommon_allocateArrayToDevice<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim);
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
        intensity = (intensity <= 255 ? reg_round(intensity) : 255); // 255=2^8-1
        return static_cast<unsigned char>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT16:
        intensity = (intensity <= 65535 ? reg_round(intensity) : 65535); // 65535=2^16-1
        return static_cast<unsigned short>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT32:
        intensity = (intensity <= 4294967295 ? reg_round(intensity) : 4294967295); // 4294967295=2^32-1
        return static_cast<unsigned int>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    default:
        return static_cast<DataType>(reg_round(intensity));
        break;
    }
}
/* *************************************************************** */
template<class T>
void CudaAladinContent::FillImageData(nifti_image *image, float *memoryObject, int type) {
    size_t size = image->nvox;
    float *buffer = (float*)malloc(size * sizeof(float));

    cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = (void*)malloc(image->nvox * image->nbyper);
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
        FillImageData<unsigned int>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT32:
        FillImageData<int>(image, memoryObject, datatype);
        break;
    default:
        std::cout << "CUDA: unsupported type" << std::endl;
        break;
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
    if (transformationMatrix != nullptr)
        cudaCommon_free(&transformationMatrix_d);

    if (reference != nullptr) {
        cudaCommon_free(&referenceImageArray_d);
        cudaCommon_free(&referenceMat_d);
    }

    if (floating != nullptr) {
        cudaCommon_free(&floatingImageArray_d);
        cudaCommon_free(&floIJKMat_d);
    }

    if (warped != nullptr)
        cudaCommon_free(&warpedImageArray_d);

    if (deformationField != nullptr)
        cudaCommon_free(&deformationFieldArray_d);

    if (referenceMask != nullptr)
        cudaCommon_free(&mask_d);

    if (blockMatchingParams != nullptr) {
        cudaCommon_free(&totalBlock_d);
        cudaCommon_free(&referencePosition_d);
        cudaCommon_free(&warpedPosition_d);
        /*
        cudaCommon_free(&AR_d);
        cudaCommon_free(&U_d);
        cudaCommon_free(&VT_d);
        cudaCommon_free(&Sigma_d);
        cudaCommon_free(&lengths_d);
        cudaCommon_free(&newWarpedPos_d);
        */
    }
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return CudaContextSingleton::Instance().GetIsCardDoubleCapable();
}
/* *************************************************************** */
