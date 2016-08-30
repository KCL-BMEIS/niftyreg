#include "CUDAF3DContent.h"

CudaF3DContent::CudaF3DContent(int refTime, int floTime) : F3DContent(NR_PLATFORM_CUDA, refTime, floTime), CudaGlobalContent(refTime, floTime), GlobalContent(NR_PLATFORM_CUDA, refTime, floTime)
{
    this->controlPointGrid_d = 0;
}
/* *************************************************************** */
CudaF3DContent::~CudaF3DContent()
{
    if (this->controlPointGrid_d != NULL) {
        cudaCommon_free<float>(&controlPointGrid_d);
    }
}
/* *************************************************************** */
float* CudaF3DContent::getControlPointGrid_d()
{
   return this->controlPointGrid_d;
}
/* *************************************************************** */
nifti_image* CudaF3DContent::getCurrentControlPointGrid(int datatype)
{
    downloadImage(this->currentControlPointGrid, this->controlPointGrid_d, datatype);
    return this->currentControlPointGrid;
}
/* *************************************************************** */
void CudaF3DContent::setCurrentControlPointGrid(nifti_image *cpgIn)
{
    if (this->currentControlPointGrid != NULL) {
        cudaCommon_free<float>(&controlPointGrid_d);
    }
    if (cpgIn->datatype != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(cpgIn);
    }
    F3DContent::setCurrentControlPointGrid(cpgIn);
    cudaCommon_allocateArrayToDevice<float>(&controlPointGrid_d, this->currentControlPointGrid->nvox);
    cudaCommon_transferFromDeviceToNiftiSimple<float>(&controlPointGrid_d, this->currentControlPointGrid);
}
/* *************************************************************** */
void CudaF3DContent::ClearControlPointGrid()
{
   if(this->currentControlPointGrid!=NULL) {
       //Already destroyed in the destructor
       //cudaCommon_free<float>(&controlPointGrid_d);
       F3DContent::ClearControlPointGrid();
   }
#ifndef NDEBUG
   reg_print_fct_debug("CudaF3DContent::ClearControlPointGrid");
#endif
}
