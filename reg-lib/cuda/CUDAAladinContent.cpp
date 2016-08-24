#include "CUDAAladinContent.h"
#include "_reg_common_cuda.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaAladinContent::CudaAladinContent() : AladinContent(NR_PLATFORM_CUDA), CudaGlobalContent(1,1), GlobalContent(NR_PLATFORM_CUDA)
{
    this->referencePosition_d = 0;
    this->warpedPosition_d = 0;
    this->totalBlock_d = 0;
    this->transformationMatrix_d = 0;
}
/* *************************************************************** */
CudaAladinContent::~CudaAladinContent()
{
   freeCuPtrs();
}
/* *************************************************************** */
void CudaAladinContent::InitBlockMatchingParams()
{
   AladinContent::InitBlockMatchingParams();
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
void CudaAladinContent::setTransformationMatrix(mat44 transformationMatrixIn)
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
void CudaAladinContent::freeCuPtrs()
{
   if (this->transformationMatrix != NULL)
      cudaCommon_free<float>(&transformationMatrix_d);

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
void CudaAladinContent::ClearBlockMatchingParams()
{
    if(this->blockMatchingParams != NULL) {
        cudaCommon_free<int>(&totalBlock_d);
        cudaCommon_free<float>(&referencePosition_d);
        cudaCommon_free<float>(&warpedPosition_d);
        AladinContent::ClearBlockMatchingParams();
    }
#ifndef NDEBUG
   reg_print_fct_debug("CudaAladinContent::ClearBlockMatchingParams");
#endif
}
