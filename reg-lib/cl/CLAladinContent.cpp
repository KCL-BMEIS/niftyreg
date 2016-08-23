#include "CLAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent() : AladinContent(NR_PLATFORM_CL), ClGlobalContent(1,1), GlobalContent(NR_PLATFORM_CL)
{
    this->referencePositionClmem = 0;
    this->warpedPositionClmem = 0;
    this->totalBlockClmem = 0;
}
/* *************************************************************** */
ClAladinContent::~ClAladinContent()
{
    freeClPtrs();
}
/* *************************************************************** */
void ClAladinContent::InitBlockMatchingParams()
{
    AladinContent::InitBlockMatchingParams();
    if (this->blockMatchingParams != NULL) {
        if (this->blockMatchingParams->referencePosition != NULL) {
            //targetPositionClmem
            clReleaseMemObject(referencePositionClmem);
            this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                                         this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
                                                                         this->blockMatchingParams->referencePosition, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (referencePositionClmem): ");
        }
        if (this->blockMatchingParams->warpedPosition != NULL) {
            //resultPositionClmem
            clReleaseMemObject(warpedPositionClmem);
            this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                                     this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
                                                                     this->blockMatchingParams->warpedPosition, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (warpedPositionClmem): ");
        }
        if (this->blockMatchingParams->totalBlock != NULL) {
            //totalBlockClmem
            clReleaseMemObject(totalBlockClmem);
            this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                this->blockMatchingParams->totalBlockNumber * sizeof(int),
                                                                this->blockMatchingParams->totalBlock, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (activeBlockClmem): ");
        }
    }
}
/* *************************************************************** */
void ClAladinContent::setTransformationMatrix(mat44 *transformationMatrixIn)
{
   AladinContent::setTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::setTransformationMatrix(mat44 transformationMatrixIn)
{
   AladinContent::setTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::setBlockMatchingParams(_reg_blockMatchingParam* bmp)
{
   AladinContent::setBlockMatchingParams(bmp);
   if (this->blockMatchingParams->referencePosition != NULL) {
      clReleaseMemObject(this->referencePositionClmem);
      //referencePositionClmem
      this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->referencePosition, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
   }
   if (this->blockMatchingParams->warpedPosition != NULL) {
      clReleaseMemObject(this->warpedPositionClmem);
      //warpedPositionClmem
      this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->warpedPosition, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
   }
   if (this->blockMatchingParams->totalBlock != NULL) {
      clReleaseMemObject(this->totalBlockClmem);
      //totalBlockClmem
      this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->totalBlockNumber * sizeof(int), this->blockMatchingParams->totalBlock, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (activeBlockClmem): ");
   }
}
/* *************************************************************** */
_reg_blockMatchingParam* ClAladinContent::getBlockMatchingParams()
{
   this->errNum = clEnqueueReadBuffer(this->commandQueue, this->warpedPositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->warpedPosition, 0, NULL, NULL); //CLCONTEXT
   this->sContext->checkErrNum(this->errNum, "CLContext: failed result position: ");
   this->errNum = clEnqueueReadBuffer(this->commandQueue, this->referencePositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->referencePosition, 0, NULL, NULL); //CLCONTEXT
   this->sContext->checkErrNum(this->errNum, "CLContext: failed target position: ");
   return this->blockMatchingParams;
}
/* *************************************************************** */
cl_mem ClAladinContent::getReferencePositionClmem()
{
   return this->referencePositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getWarpedPositionClmem()
{
   return this->warpedPositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getTotalBlockClmem()
{
   return this->totalBlockClmem;
}
/* *************************************************************** */
void ClAladinContent::freeClPtrs()
{
    if(this->blockMatchingParams != NULL) {
		clReleaseMemObject(this->totalBlockClmem);
		clReleaseMemObject(this->referencePositionClmem);
		clReleaseMemObject(this->warpedPositionClmem);
	}
}
/* *************************************************************** */
void ClAladinContent::ClearBlockMatchingParams()
{
    if(this->blockMatchingParams != NULL) {
        clReleaseMemObject(this->totalBlockClmem);
        clReleaseMemObject(this->referencePositionClmem);
        clReleaseMemObject(this->warpedPositionClmem);
        AladinContent::ClearBlockMatchingParams();
    }
#ifndef NDEBUG
   reg_print_fct_debug("ClAladinContent::ClearBlockMatchingParams");
#endif
}
/* *************************************************************** */
