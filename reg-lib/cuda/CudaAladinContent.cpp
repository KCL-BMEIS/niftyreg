#include "CudaAladinContent.h"
#include "_reg_common_cuda.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent() {
    InitVars();
    AllocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *currentReferenceIn,
                                     nifti_image *currentFloatingIn,
                                     int *currentReferenceMaskIn,
                                     size_t byte,
                                     const unsigned int blockPercentage,
                                     const unsigned int inlierLts,
                                     int blockStep) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  sizeof(float), // forcing float for CUDA
                  blockPercentage,
                  inlierLts,
                  blockStep) {
    if (byte != sizeof(float)) {
        reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
        reg_print_msg_warn("Datatype has been forced to float");
    }
    InitVars();
    AllocateCuPtrs();

}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *currentReferenceIn,
                                     nifti_image *currentFloatingIn,
                                     int *currentReferenceMaskIn,
                                     size_t byte) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  sizeof(float)) // forcing float for CUDA
{
    if (byte != sizeof(float)) {
        reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
        reg_print_msg_warn("Datatype has been forced to float");
    }
    InitVars();
    AllocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *currentReferenceIn,
                                     nifti_image *currentFloatingIn,
                                     int *currentReferenceMaskIn,
                                     mat44 *transMat,
                                     size_t byte,
                                     const unsigned int blockPercentage,
                                     const unsigned int inlierLts,
                                     int blockStep) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  transMat,
                  sizeof(float), // forcing float for CUDA
                  blockPercentage,
                  inlierLts,
                  blockStep) {
    if (byte != sizeof(float)) {
        reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
        reg_print_msg_warn("Datatype has been forced to float");
    }
    InitVars();
    AllocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *currentReferenceIn,
                                     nifti_image *currentFloatingIn,
                                     int *currentReferenceMaskIn,
                                     mat44 *transMat,
                                     size_t byte) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  transMat,
                  sizeof(float)) // forcing float for CUDA
{
    if (byte != sizeof(float)) {
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
    this->referenceImageArray_d = 0;
    this->floatingImageArray_d = 0;
    this->warpedImageArray_d = 0;
    this->deformationFieldArray_d = 0;
    this->referencePosition_d = 0;
    this->warpedPosition_d = 0;
    this->totalBlock_d = 0;
    this->mask_d = 0;
    this->floIJKMat_d = 0;

    if (this->currentReference != nullptr && this->currentReference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(this->currentReference);
    if (this->currentFloating != nullptr && this->currentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(this->currentFloating);
        if (this->currentWarped != nullptr)
            reg_tools_changeDatatype<float>(this->currentWarped);
    }

    this->cudaSContext = &CudaContextSingleton::Instance();
    this->cudaContext = this->cudaSContext->GetContext();

    //this->numBlocks = (this->blockMatchingParams->activeBlock != nullptr) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void CudaAladinContent::AllocateCuPtrs() {
    if (this->transformationMatrix != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);

        float *tmpMat_h = (float*)malloc(16 * sizeof(float));
        mat44ToCptr(*(this->transformationMatrix), tmpMat_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));

        free(tmpMat_h);
    }
    if (this->currentReferenceMask != nullptr) {
        cudaCommon_allocateArrayToDevice<int>(&mask_d, currentReference->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->currentReferenceMask, currentReference->nvox);
    }
    if (this->currentReference != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, currentReference->nvox);
        cudaCommon_allocateArrayToDevice<float>(&referenceMat_d, 16);

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->currentReference);

        float* targetMat = (float *)malloc(16 * sizeof(float)); //freed
        mat44ToCptr(this->refMatrix_xyz, targetMat);
        cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, 16);
        free(targetMat);
    }
    if (this->currentWarped != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->currentWarped->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->currentWarped);
    }
    if (this->currentDeformationField != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->currentDeformationField->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->currentDeformationField);
    }
    if (this->currentFloating != nullptr) {
        cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, this->currentFloating->nvox);
        cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, 16);

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->currentFloating);

        float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
        mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
        free(sourceIJKMatrix_h);
    }

    if (this->blockMatchingParams != nullptr) {
        if (this->blockMatchingParams->referencePosition != nullptr) {
            cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        }
        if (this->blockMatchingParams->warpedPosition != nullptr) {
            cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        }
        if (this->blockMatchingParams->totalBlock != nullptr) {
            cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
            cudaCommon_transferFromDeviceToNiftiSimple1<int>(&totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
        /* // Removed until CUDA SVD is added back
        if (this->blockMatchingParams->activeBlockNumber > 0 ) {
           unsigned int m = blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim;
           unsigned int n = 0;

           if (this->blockMatchingParams->dim == 2) {
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
           cudaCommon_allocateArrayToDevice<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        }
        */
    }
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetCurrentWarped(int type) {
    DownloadImage(currentWarped, warpedImageArray_d, type);
    return currentWarped;
}
/* *************************************************************** */
nifti_image* CudaAladinContent::GetCurrentDeformationField() {

    cudaCommon_transferFromDeviceToCpu<float>((float*)currentDeformationField->data, &deformationFieldArray_d, currentDeformationField->nvox);
    return currentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::GetBlockMatchingParams() {

    cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->warpedPosition, &warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->referencePosition, &referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    return this->blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    if (this->transformationMatrix != nullptr)
        cudaCommon_free<float>(&transformationMatrix_d);

    AladinContent::SetTransformationMatrix(transformationMatrixIn);
    float *tmpMat_h = (float*)malloc(16 * sizeof(float));
    mat44ToCptr(*(this->transformationMatrix), tmpMat_h);

    cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
    free(tmpMat_h);
}
/* *************************************************************** */
void CudaAladinContent::SetCurrentDeformationField(nifti_image *currentDeformationFieldIn) {
    if (this->currentDeformationField != nullptr)
        cudaCommon_free<float>(&deformationFieldArray_d);
    AladinContent::SetCurrentDeformationField(currentDeformationFieldIn);

    cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->currentDeformationField->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->currentDeformationField);
}
/* *************************************************************** */
void CudaAladinContent::SetCurrentReferenceMask(int *maskIn, size_t nvox) {
    if (this->currentReferenceMask != nullptr)
        cudaCommon_free<int>(&mask_d);
    this->currentReferenceMask = maskIn;
    cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
    cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}
/* *************************************************************** */
void CudaAladinContent::SetCurrentWarped(nifti_image *currentWarped) {
    if (this->currentWarped != nullptr)
        cudaCommon_free<float>(&warpedImageArray_d);
    AladinContent::SetCurrentWarped(currentWarped);
    reg_tools_changeDatatype<float>(this->currentWarped);

    cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, currentWarped->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->currentWarped);
}
/* *************************************************************** */
void CudaAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (this->blockMatchingParams->referencePosition != nullptr) {
        cudaCommon_free<float>(&referencePosition_d);
        //referencePosition
        cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    }
    if (this->blockMatchingParams->warpedPosition != nullptr) {
        cudaCommon_free<float>(&warpedPosition_d);
        //warpedPosition
        cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    }
    if (this->blockMatchingParams->totalBlock != nullptr) {
        cudaCommon_free<int>(&totalBlock_d);
        //activeBlock
        cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, this->blockMatchingParams->totalBlockNumber);
        cudaCommon_transferArrayFromCpuToDevice<int>(&totalBlock_d, this->blockMatchingParams->totalBlock, this->blockMatchingParams->totalBlockNumber);
    }
    /* // Removed until CUDA SVD is added back
     if (this->blockMatchingParams->activeBlockNumber > 0) {
         unsigned int m = blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim;
         unsigned int n = 0;

         if (this->blockMatchingParams->dim == 2) {
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
         cudaCommon_allocateArrayToDevice<float>(&newWarpedPos_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
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
void CudaAladinContent::FillImageData(nifti_image *image,
                                      float* memoryObject,
                                      int type) {
    size_t size = image->nvox;
    float* buffer = nullptr;
    buffer = (float*)malloc(size * sizeof(float));

    if (buffer == nullptr) {
        reg_print_fct_error("\nERROR: Memory allocation did not complete successfully!");
    }

    cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = (void *)malloc(image->nvox * image->nbyper);
    T* dataT = static_cast<T*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = FillWarpedImageData<T>(buffer[i], type);
    free(buffer);
}
/* *************************************************************** */
void CudaAladinContent::DownloadImage(nifti_image *image,
                                      float* memoryObject,
                                      int datatype) {
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
    if (this->transformationMatrix != nullptr)
        cudaCommon_free<float>(&transformationMatrix_d);

    if (this->currentReference != nullptr) {
        cudaCommon_free<float>(&referenceImageArray_d);
        cudaCommon_free<float>(&referenceMat_d);
    }

    if (this->currentFloating != nullptr) {
        cudaCommon_free<float>(&floatingImageArray_d);
        cudaCommon_free<float>(&floIJKMat_d);
    }

    if (this->currentWarped != nullptr)
        cudaCommon_free<float>(&warpedImageArray_d);

    if (this->currentDeformationField != nullptr)
        cudaCommon_free<float>(&deformationFieldArray_d);

    if (this->currentReferenceMask != nullptr)
        cudaCommon_free<int>(&mask_d);

    if (this->blockMatchingParams != nullptr) {
        cudaCommon_free<int>(&totalBlock_d);
        cudaCommon_free<float>(&referencePosition_d);
        cudaCommon_free<float>(&warpedPosition_d);
        /*
        cudaCommon_free<float>(&AR_d);
        cudaCommon_free<float>(&U_d);
        cudaCommon_free<float>(&VT_d);
        cudaCommon_free<float>(&Sigma_d);
        cudaCommon_free<float>(&lengths_d);
        cudaCommon_free<float>(&newWarpedPos_d);
        */
    }
}
/* *************************************************************** */
bool CudaAladinContent::IsCurrentComputationDoubleCapable() {
    return this->cudaSContext->GetIsCardDoubleCapable();
}
/* *************************************************************** */
