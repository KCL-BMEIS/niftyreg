#include "ClF3DContent.h"

ClF3DContent::ClF3DContent(int refTime, int floTime) : F3DContent(NR_PLATFORM_CL, refTime, floTime), ClGlobalContent(refTime, floTime), GlobalContent(NR_PLATFORM_CL, refTime, floTime)
{
    this->controlPointGridClmem = 0;
}
/* *************************************************************** */
ClF3DContent::~ClF3DContent()
{
    if(this->currentControlPointGrid != NULL) {
        clReleaseMemObject(this->controlPointGridClmem);
    }
}
/* *************************************************************** */
cl_mem ClF3DContent::getControlPointGridClmem()
{
   return this->controlPointGridClmem;
}
/* *************************************************************** */
nifti_image* ClF3DContent::getCurrentControlPointGrid(int datatype)
{
    downloadImage(this->currentControlPointGrid, this->controlPointGridClmem, datatype);
    return this->currentControlPointGrid;
}
/* *************************************************************** */
void ClF3DContent::setCurrentControlPointGrid(nifti_image *cpgIn)
{
    if (this->currentControlPointGrid != NULL) {
                clReleaseMemObject(this->controlPointGridClmem);
    }
    if (cpgIn->datatype != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(cpgIn);
    }
    F3DContent::setCurrentControlPointGrid(cpgIn);
    this->controlPointGridClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->currentControlPointGrid->nvox * sizeof(float), this->currentControlPointGrid->data, &this->errNum);
    this->sContext->checkErrNum(this->errNum, "ClGlobalContent::setCurrentWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClF3DContent::ClearControlPointGrid()
{
   if(this->currentControlPointGrid!=NULL) {
       clReleaseMemObject(this->controlPointGridClmem);
       F3DContent::ClearControlPointGrid();
   }
#ifndef NDEBUG
   reg_print_fct_debug("ClGlobalContent::ClearWarped");
#endif
}
