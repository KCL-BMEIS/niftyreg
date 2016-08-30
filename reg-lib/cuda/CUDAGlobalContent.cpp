#include "CUDAGlobalContent.h"

/* *************************************************************** */
CudaGlobalContent::CudaGlobalContent(int refTimePoint,int floTimePoint) : GlobalContent(NR_PLATFORM_CUDA, refTimePoint, floTimePoint)
{
    this->referenceImageArray_d = 0;
    this->floatingImageArray_d = 0;
    this->warpedImageArray_d = 0;
    this->deformationFieldArray_d = 0;
    this->mask_d = 0;

    this->referenceMat_d = 0;
    this->floIJKMat_d = 0;

    this->cudaSContext = &CUDAContextSingletton::Instance();
    this->cudaContext = this->cudaSContext->getContext();
}
/* *************************************************************** */
CudaGlobalContent::~CudaGlobalContent()
{
   freeCuPtrs();
}
/* *************************************************************** */
void CudaGlobalContent::AllocateWarped()
{
    GlobalContent::AllocateWarped();
    if(this->currentWarped != NULL) {
       cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->currentWarped->nvox);
       cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->currentWarped);
    }
}
/* *************************************************************** */
void CudaGlobalContent::AllocateDeformationField()
{
    GlobalContent::AllocateDeformationField();
    if (this->currentDeformationField != NULL) {
       cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->currentDeformationField->nvox);
       cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->currentDeformationField);
    }
}
/* *************************************************************** */
nifti_image *CudaGlobalContent::getCurrentWarped(int type)
{
   downloadImage(this->currentWarped, warpedImageArray_d, type);
   return this->currentWarped;
}
/* *************************************************************** */
nifti_image *CudaGlobalContent::getCurrentDeformationField()
{
   cudaCommon_transferFromDeviceToCpu<float>((float*) this->currentDeformationField->data, &deformationFieldArray_d, this->currentDeformationField->nvox);
   return this->currentDeformationField;
}
/* *************************************************************** */
void CudaGlobalContent::setCurrentReference(nifti_image *currentRefIn)
{
    if (this->currentReference != NULL) {
       cudaCommon_free<float>(&referenceImageArray_d);
       cudaCommon_free<float>(&referenceMat_d);
    }

    GlobalContent::setCurrentReference(currentRefIn);
    cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, this->currentReference->nvox);
    cudaCommon_allocateArrayToDevice<float>(&referenceMat_d, 16);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->currentReference);

    float* targetMat = (float *)malloc(16 * sizeof(float)); //freed
    mat44ToCptr(*this->refMatrix_xyz, targetMat);
    cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, 16);
    free(targetMat);
}
/* *************************************************************** */
void CudaGlobalContent::setCurrentReferenceMask(int *maskIn, size_t nvox)
{
   if (this->currentReferenceMask != NULL)
      cudaCommon_free<int>(&mask_d);

   GlobalContent::setCurrentReferenceMask(maskIn, nvox);
   cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
   cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}
/* *************************************************************** */
void CudaGlobalContent::setCurrentFloating(nifti_image *currentFloIn)
{
    if (this->currentFloating != NULL) {
        cudaCommon_free<float>(&floatingImageArray_d);
        cudaCommon_free<float>(&floIJKMat_d);
    }

    GlobalContent::setCurrentFloating(currentFloIn);
    cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, this->currentFloating->nvox);
    cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, 16);

    cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->currentFloating);

    float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
    mat44ToCptr(*this->floMatrix_ijk, sourceIJKMatrix_h);
    NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
    free(sourceIJKMatrix_h);
}
/* *************************************************************** */
void CudaGlobalContent::setCurrentWarped(nifti_image *currentWarpedIn)
{
   if (this->currentWarped != NULL) {
      cudaCommon_free<float>(&warpedImageArray_d);
   }

   GlobalContent::setCurrentWarped(currentWarpedIn);
   cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->currentWarped->nvox);
   cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->currentWarped);
}
/* *************************************************************** */
void CudaGlobalContent::setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
{
   if (this->currentDeformationField != NULL)
      cudaCommon_free<float>(&deformationFieldArray_d);

   GlobalContent::setCurrentDeformationField(CurrentDeformationFieldIn);
   cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->currentDeformationField->nvox);
   cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->currentDeformationField);
}
/* *************************************************************** */
template<class DataType>
DataType CudaGlobalContent::fillWarpedImageData(float intensity, int datatype) {

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
void CudaGlobalContent::fillImageData(nifti_image *image,
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
void CudaGlobalContent::downloadImage(nifti_image *image,
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
float* CudaGlobalContent::getReferenceImageArray_d()
{
   return referenceImageArray_d;
}
/* *************************************************************** */
float* CudaGlobalContent::getFloatingImageArray_d()
{
   return floatingImageArray_d;
}
/* *************************************************************** */
float* CudaGlobalContent::getWarpedImageArray_d()
{
   return warpedImageArray_d;
}
/* *************************************************************** */
float* CudaGlobalContent::getDeformationFieldArray_d()
{
   return deformationFieldArray_d;
}
/* *************************************************************** */
float* CudaGlobalContent::getReferenceMat_d()
{
   return referenceMat_d;
}
/* *************************************************************** */
float* CudaGlobalContent::getFloIJKMat_d()
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
int *CudaGlobalContent::getMask_d()
{
   return mask_d;
}
/* *************************************************************** */
int *CudaGlobalContent::getReferenceDims()
{
   return referenceDims;
}
/* *************************************************************** */
int *CudaGlobalContent::getFloatingDims()
{
   return floatingDims;
}
/* *************************************************************** */
void CudaGlobalContent::freeCuPtrs()
{
   if (this->currentReference != NULL) {
      cudaCommon_free<float>(&referenceImageArray_d);
      cudaCommon_free<float>(&referenceMat_d);
   }

   if (this->currentFloating != NULL) {
      cudaCommon_free<float>(&floatingImageArray_d);
      cudaCommon_free<float>(&floIJKMat_d);
   }

   if (this->currentWarped != NULL)
      cudaCommon_free<float>(&warpedImageArray_d);

   if (this->currentDeformationField != NULL)
      cudaCommon_free<float>(&deformationFieldArray_d);

   if (this->currentReferenceMask != NULL)
      cudaCommon_free<int>(&mask_d);

      /*
      cudaCommon_free<float>(&AR_d);
      cudaCommon_free<float>(&U_d);
      cudaCommon_free<float>(&VT_d);
      cudaCommon_free<float>(&Sigma_d);
      cudaCommon_free<float>(&lengths_d);
      cudaCommon_free<float>(&newWarpedPos_d);
      */
}
/* *************************************************************** */
void CudaGlobalContent::ClearWarped()
{
    if (this->currentWarped != NULL) {
       cudaCommon_free<float>(&warpedImageArray_d);
       GlobalContent::ClearWarped();
    }
}
/* *************************************************************** */
void CudaGlobalContent::ClearDeformationField()
{
    if (this->currentDeformationField != NULL) {
       cudaCommon_free<float>(&deformationFieldArray_d);
       GlobalContent::ClearDeformationField();
    }
}
/* *************************************************************** */
bool CudaGlobalContent::isCurrentComputationDoubleCapable() {
    return this->cudaSContext->getIsCardDoubleCapable();
}
/* *************************************************************** */
