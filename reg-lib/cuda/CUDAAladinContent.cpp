#include "CUDAAladinContent.h"
#include "_reg_common_cuda.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent()
{
   initVars();
   allocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *CurrentReferenceIn,
                         nifti_image *CurrentFloatingIn,
                         int *CurrentReferenceMaskIn,
                         size_t byte,
                         const unsigned int blockPercentage,
                         const unsigned int inlierLts,
                         int blockStep) :
   AladinContent(CurrentReferenceIn,
           CurrentFloatingIn,
           CurrentReferenceMaskIn,
           sizeof(float), // forcing float for CUDA
           blockPercentage,
           inlierLts,
           blockStep)
{
   if(byte!=sizeof(float)){
      reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
      reg_print_msg_warn("Datatype has been forced to float");
   }
   initVars();
   allocateCuPtrs();

}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *CurrentReferenceIn,
                         nifti_image *CurrentFloatingIn,
                         int *CurrentReferenceMaskIn,
                         size_t byte) :
   AladinContent(CurrentReferenceIn,
           CurrentFloatingIn,
           CurrentReferenceMaskIn,
           sizeof(float)) // forcing float for CUDA
{
   if(byte!=sizeof(float)){
      reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
      reg_print_msg_warn("Datatype has been forced to float");
   }
   initVars();
   allocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *CurrentReferenceIn,
                         nifti_image *CurrentFloatingIn,
                         int *CurrentReferenceMaskIn,
                         mat44 *transMat,
                         size_t byte,
                         const unsigned int blockPercentage,
                         const unsigned int inlierLts,
                         int blockStep) :
   AladinContent(CurrentReferenceIn,
           CurrentFloatingIn,
           CurrentReferenceMaskIn,
           transMat,
           sizeof(float), // forcing float for CUDA
           blockPercentage,
           inlierLts,
           blockStep)
{
   if(byte!=sizeof(float)){
      reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
      reg_print_msg_warn("Datatype has been forced to float");
   }
   initVars();
   allocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::CudaAladinContent(nifti_image *CurrentReferenceIn,
                         nifti_image *CurrentFloatingIn,
                         int *CurrentReferenceMaskIn,
                         mat44 *transMat,
                         size_t byte) :
   AladinContent(CurrentReferenceIn,
           CurrentFloatingIn,
           CurrentReferenceMaskIn,
           transMat,
           sizeof(float)) // forcing float for CUDA
{
   if(byte!=sizeof(float)){
      reg_print_fct_warn("CudaAladinContent::CudaAladinContent");
      reg_print_msg_warn("Datatype has been forced to float");
   }
   initVars();
   allocateCuPtrs();
}
/* *************************************************************** */
CudaAladinContent::~CudaAladinContent()
{
   freeCuPtrs();
}
/* *************************************************************** */
void CudaAladinContent::initVars()
{
   this->referenceImageArray_d = 0;
   this->floatingImageArray_d = 0;
   this->warpedImageArray_d = 0;
   this->deformationFieldArray_d = 0;
   this->referencePosition_d = 0;
   this->warpedPosition_d = 0;
   this->totalBlock_d = 0;
   this->mask_d = 0;
   this->floIJKMat_d = 0;

   if (this->CurrentReference != NULL && this->CurrentReference->nbyper != NIFTI_TYPE_FLOAT32)
      reg_tools_changeDatatype<float>(this->CurrentReference);
   if (this->CurrentFloating != NULL && this->CurrentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
      reg_tools_changeDatatype<float>(this->CurrentFloating);
      if (this->CurrentWarped != NULL)
         reg_tools_changeDatatype<float>(this->CurrentWarped);
   }

   this->cudaSContext = &CUDAContextSingletton::Instance();
   this->cudaContext = this->cudaSContext->getContext();

   this->referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
   this->floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
   //this->numBlocks = (this->blockMatchingParams->activeBlock != NULL) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void CudaAladinContent::allocateCuPtrs()
{

   if (this->transformationMatrix != NULL) {
      cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);

      float *tmpMat_h = (float*)malloc(16 * sizeof(float));
      mat44ToCptr(*(this->transformationMatrix), tmpMat_h);
      NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));

      free(tmpMat_h);
   }
   if (this->CurrentReferenceMask != NULL) {
      cudaCommon_allocateArrayToDevice<int>(&mask_d, this->referenceVoxels);
      cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels);
   }
   if (this->CurrentReference != NULL) {
      cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceVoxels);
      cudaCommon_allocateArrayToDevice<float>(&referenceMat_d, 16);

      cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference);

      float* targetMat = (float *)malloc(16 * sizeof(float)); //freed
      mat44ToCptr(this->refMatrix_xyz, targetMat);
      cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, 16);
      free(targetMat);
   }
   if (this->CurrentWarped != NULL) {
      cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->CurrentWarped->nvox);
      cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
   }
   if (this->CurrentDeformationField != NULL) {
      cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
      cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
   }
   if (this->CurrentFloating != NULL) {
      cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingVoxels);
      cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, 16);

      cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->CurrentFloating);

      float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
      mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
      NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
      free(sourceIJKMatrix_h);
   }

   if (this->blockMatchingParams != NULL) {
      if (this->blockMatchingParams->referencePosition != NULL) {
         cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
         cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
      }
      if (this->blockMatchingParams->warpedPosition != NULL) {
         cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
         cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
      }
      if (this->blockMatchingParams->totalBlock != NULL) {
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
nifti_image *CudaAladinContent::getCurrentWarped(int type)
{
   downloadImage(CurrentWarped, warpedImageArray_d, type);
   return CurrentWarped;
}
/* *************************************************************** */
nifti_image *CudaAladinContent::getCurrentDeformationField()
{

   cudaCommon_transferFromDeviceToCpu<float>((float*) CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
   return CurrentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaAladinContent::getBlockMatchingParams()
{

   cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->warpedPosition, &warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
   cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->referencePosition, &referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
   return this->blockMatchingParams;
}
/* *************************************************************** */
void CudaAladinContent::setTransformationMatrix(mat44 *transformationMatrixIn)
{
   if (this->transformationMatrix != NULL)
      cudaCommon_free<float>(&transformationMatrix_d);

   AladinContent::setTransformationMatrix(transformationMatrixIn);
   float *tmpMat_h = (float*)malloc(16 * sizeof(float));
   mat44ToCptr(*(this->transformationMatrix), tmpMat_h);

   cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
   NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
   free(tmpMat_h);
}
/* *************************************************************** */
void CudaAladinContent::setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
{
   if (this->CurrentDeformationField != NULL)
      cudaCommon_free<float>(&deformationFieldArray_d);
   AladinContent::setCurrentDeformationField(CurrentDeformationFieldIn);

   cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
   cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
}
/* *************************************************************** */
void CudaAladinContent::setCurrentReferenceMask(int *maskIn, size_t nvox)
{
   if (this->CurrentReferenceMask != NULL)
      cudaCommon_free<int>(&mask_d);
   this->CurrentReferenceMask = maskIn;
   cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
   cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}
/* *************************************************************** */
void CudaAladinContent::setCurrentWarped(nifti_image *currentWarped)
{
   if (this->CurrentWarped != NULL)
      cudaCommon_free<float>(&warpedImageArray_d);
   AladinContent::setCurrentWarped(currentWarped);
   reg_tools_changeDatatype<float>(this->CurrentWarped);

   cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, CurrentWarped->nvox);
   cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
}
/* *************************************************************** */
void CudaAladinContent::setBlockMatchingParams(_reg_blockMatchingParam* bmp)
{

   AladinContent::setBlockMatchingParams(bmp);
   if (this->blockMatchingParams->referencePosition != NULL) {
      cudaCommon_free<float>(&referencePosition_d);
      //referencePosition
      cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
      cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
   }
   if (this->blockMatchingParams->warpedPosition != NULL) {
      cudaCommon_free<float>(&warpedPosition_d);
      //warpedPosition
      cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
      cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
   }
   if (this->blockMatchingParams->totalBlock != NULL) {
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
DataType CudaAladinContent::fillWarpedImageData(float intensity, int datatype) {

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
void CudaAladinContent::fillImageData(nifti_image *image,
                                float* memoryObject,
                                int type)
{

   size_t size = image->nvox;
   float* buffer = NULL;
   buffer = (float*) malloc(size * sizeof(float));

   if (buffer == NULL) {
      reg_print_fct_error("\nERROR: Memory allocation did not complete successfully!");
   }

   cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

   free(image->data);
   image->datatype = type;
   image->nbyper = sizeof(T);
   image->data = (void *)malloc(image->nvox*image->nbyper);
   T* dataT = static_cast<T*>(image->data);
   for (size_t i = 0; i < size; ++i)
       dataT[i] = fillWarpedImageData<T>(buffer[i], type);
   free(buffer);
}
/* *************************************************************** */
void CudaAladinContent::downloadImage(nifti_image *image,
                                float* memoryObject,
                                int datatype)
{
   switch (datatype) {
   case NIFTI_TYPE_FLOAT32:
      fillImageData<float>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_FLOAT64:
      fillImageData<double>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_UINT8:
      fillImageData<unsigned char>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_INT8:
      fillImageData<char>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_UINT16:
      fillImageData<unsigned short>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_INT16:
      fillImageData<short>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_UINT32:
      fillImageData<unsigned int>(image, memoryObject, datatype);
      break;
   case NIFTI_TYPE_INT32:
      fillImageData<int>(image, memoryObject, datatype);
      break;
   default:
      std::cout << "CUDA: unsupported type" << std::endl;
      break;
   }
}
/* *************************************************************** */
float* CudaAladinContent::getReferenceImageArray_d()
{
   return referenceImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::getFloatingImageArray_d()
{
   return floatingImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::getWarpedImageArray_d()
{
   return warpedImageArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::getTransformationMatrix_d()
{
   return transformationMatrix_d;
}
/* *************************************************************** */
float* CudaAladinContent::getReferencePosition_d()
{
   return referencePosition_d;
}
/* *************************************************************** */
float* CudaAladinContent::getWarpedPosition_d()
{
   return warpedPosition_d;
}
/* *************************************************************** */
float* CudaAladinContent::getDeformationFieldArray_d()
{
   return deformationFieldArray_d;
}
/* *************************************************************** */
float* CudaAladinContent::getReferenceMat_d()
{
   return referenceMat_d;
}
/* *************************************************************** */
float* CudaAladinContent::getFloIJKMat_d()
{
   return floIJKMat_d;
}
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getAR_d()
{
   return AR_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getU_d()
{
   return U_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getVT_d()
{
   return VT_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getSigma_d()
{
   return Sigma_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getLengths_d()
{
   return lengths_d;
}
*/
/* *************************************************************** */
/* // Removed until CUDA SVD is added back
float* CudaAladinContent::getNewWarpedPos_d()
{
   return newWarpedPos_d;
}
*/
/* *************************************************************** */
int *CudaAladinContent::getTotalBlock_d()
{
   return totalBlock_d;
}
/* *************************************************************** */
int *CudaAladinContent::getMask_d()
{
   return mask_d;
}
/* *************************************************************** */
int *CudaAladinContent::getReferenceDims()
{
   return referenceDims;
}
/* *************************************************************** */
int *CudaAladinContent::getFloatingDims()
{
   return floatingDims;
}
/* *************************************************************** */
void CudaAladinContent::freeCuPtrs()
{
   if (this->transformationMatrix != NULL)
      cudaCommon_free<float>(&transformationMatrix_d);

   if (this->CurrentReference != NULL) {
      cudaCommon_free<float>(&referenceImageArray_d);
      cudaCommon_free<float>(&referenceMat_d);
   }

   if (this->CurrentFloating != NULL) {
      cudaCommon_free<float>(&floatingImageArray_d);
      cudaCommon_free<float>(&floIJKMat_d);
   }

   if (this->CurrentWarped != NULL)
      cudaCommon_free<float>(&warpedImageArray_d);

   if (this->CurrentDeformationField != NULL)
      cudaCommon_free<float>(&deformationFieldArray_d);

   if (this->CurrentReferenceMask != NULL)
      cudaCommon_free<int>(&mask_d);

   if (this->blockMatchingParams != NULL) {
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
bool CudaAladinContent::isCurrentComputationDoubleCapable() {
    return this->cudaSContext->getIsCardDoubleCapable();
}
/* *************************************************************** */
