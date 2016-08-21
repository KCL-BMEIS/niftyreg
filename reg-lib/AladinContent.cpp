#include "AladinContent.h"

using namespace std;

/* *************************************************************** */
AladinContent::AladinContent(int platformCodeIn) : GlobalContent(platformCodeIn)
{
	//int dim[8] = { 2, 20, 20, 1, 1, 1, 1, 1 };
	//this->CurrentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	//this->CurrentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	//this->CurrentReferenceMask = NULL;
    //
    this->transformationMatrix = new mat44;
    this->blockMatchingParams = new _reg_blockMatchingParam();
	this->bytes = sizeof(float);//Default
	//
}
/* *************************************************************** */
AladinContent::~AladinContent()
{
   if(this->transformationMatrix != NULL) {
       delete this->transformationMatrix;
       this->transformationMatrix = NULL;
   }
   ClearBlockMatchingParams();
#ifndef NDEBUG
   reg_print_fct_debug("AladinContent::~AladinContent()");
#endif
}
/* *************************************************************** */
void AladinContent::InitBlockMatchingParams()
{
   if (this->blockMatchingParams != NULL
    && this->currentReference != NULL
    && this->currentReferenceMask != NULL) {
      initialise_block_matching_method(currentReference,
                                       blockMatchingParams,
                                       currentReferenceMask,
                                       false);
   }
#ifndef NDEBUG
    if(this->currentReference==NULL) reg_print_msg_debug("currentReference image is NULL");
    if(this->currentFloating==NULL) reg_print_msg_debug("currentFloating image is NULL");
    if(this->currentDeformationField==NULL) reg_print_msg_debug("currentDeformationField image is NULL");
    if(this->currentWarped==NULL) reg_print_msg_debug("currentWarped image is NULL");
    if(this->currentReferenceMask==NULL) reg_print_msg_debug("currentReferenceMask image is NULL");
	if(this->blockMatchingParams==NULL) reg_print_msg_debug("blockMatchingParams image is NULL");
#endif
}
/* *************************************************************** */
void AladinContent::setCaptureRange(const int voxelCaptureRangeIn)
{
	this->blockMatchingParams->voxelCaptureRange = voxelCaptureRangeIn;
}
/* *************************************************************** */
void AladinContent::setPercentageOfBlock(unsigned pob)
{
    this->blockMatchingParams->percent_to_keep_block=pob;
}
/* *************************************************************** */
unsigned AladinContent::getPercentageOfBlock()
{
    return this->blockMatchingParams->percent_to_keep_block;
}
/* *************************************************************** */
void AladinContent::setInlierLts(unsigned ilts)
{
    this->blockMatchingParams->percent_to_keep_opt=ilts;
}
/* *************************************************************** */
unsigned AladinContent::getInlierLts()
{
    return this->blockMatchingParams->percent_to_keep_opt;
}
/* *************************************************************** */
void AladinContent::setBlockStepSize(int bss)
{
    this->blockMatchingParams->stepSize=bss;
}
/* *************************************************************** */
int AladinContent::getBlockStepSize()
{
    return this->blockMatchingParams->stepSize;
}
/* *************************************************************** */
void AladinContent::setBlockMatchingParams(_reg_blockMatchingParam* bmp)
{
    AladinContent::ClearBlockMatchingParams();
    this->blockMatchingParams = bmp;
}
/* *************************************************************** */
_reg_blockMatchingParam* AladinContent::getBlockMatchingParams()
{
    return this->blockMatchingParams;
}
/* *************************************************************** */
void AladinContent::setTransformationMatrix(mat44 *transformationMatrixIn)
{
    //if (this->transformationMatrix != NULL) {
    //   delete this->transformationMatrix;
    //   this->transformationMatrix = NULL;
    //}
    this->transformationMatrix->m[0][0] = transformationMatrixIn->m[0][0];
    this->transformationMatrix->m[0][1] = transformationMatrixIn->m[0][1];
    this->transformationMatrix->m[0][2] = transformationMatrixIn->m[0][2];
    this->transformationMatrix->m[0][3] = transformationMatrixIn->m[0][3];
    this->transformationMatrix->m[1][0] = transformationMatrixIn->m[1][0];
    this->transformationMatrix->m[1][1] = transformationMatrixIn->m[1][1];
    this->transformationMatrix->m[1][2] = transformationMatrixIn->m[1][2];
    this->transformationMatrix->m[1][3] = transformationMatrixIn->m[1][3];
    this->transformationMatrix->m[2][0] = transformationMatrixIn->m[2][0];
    this->transformationMatrix->m[2][1] = transformationMatrixIn->m[2][1];
    this->transformationMatrix->m[2][2] = transformationMatrixIn->m[2][2];
    this->transformationMatrix->m[2][3] = transformationMatrixIn->m[2][3];
    this->transformationMatrix->m[3][0] = transformationMatrixIn->m[3][0];
    this->transformationMatrix->m[3][1] = transformationMatrixIn->m[3][1];
    this->transformationMatrix->m[3][2] = transformationMatrixIn->m[3][2];
    this->transformationMatrix->m[3][3] = transformationMatrixIn->m[3][3];
}
/* *************************************************************** */
void AladinContent::setTransformationMatrix(mat44 transformationMatrixIn)
{
    //if (this->transformationMatrix != NULL) {
    //   delete this->transformationMatrix;
    //   this->transformationMatrix = NULL;
    //}
    this->transformationMatrix->m[0][0] = transformationMatrixIn.m[0][0];
    this->transformationMatrix->m[0][1] = transformationMatrixIn.m[0][1];
    this->transformationMatrix->m[0][2] = transformationMatrixIn.m[0][2];
    this->transformationMatrix->m[0][3] = transformationMatrixIn.m[0][3];
    this->transformationMatrix->m[1][0] = transformationMatrixIn.m[1][0];
    this->transformationMatrix->m[1][1] = transformationMatrixIn.m[1][1];
    this->transformationMatrix->m[1][2] = transformationMatrixIn.m[1][2];
    this->transformationMatrix->m[1][3] = transformationMatrixIn.m[1][3];
    this->transformationMatrix->m[2][0] = transformationMatrixIn.m[2][0];
    this->transformationMatrix->m[2][1] = transformationMatrixIn.m[2][1];
    this->transformationMatrix->m[2][2] = transformationMatrixIn.m[2][2];
    this->transformationMatrix->m[2][3] = transformationMatrixIn.m[2][3];
    this->transformationMatrix->m[3][0] = transformationMatrixIn.m[3][0];
    this->transformationMatrix->m[3][1] = transformationMatrixIn.m[3][1];
    this->transformationMatrix->m[3][2] = transformationMatrixIn.m[3][2];
    this->transformationMatrix->m[3][3] = transformationMatrixIn.m[3][3];
}
/* *************************************************************** */
mat44* AladinContent::getTransformationMatrix()
{
    return this->transformationMatrix;
}
/* *************************************************************** */
void AladinContent::ClearBlockMatchingParams()
{
    if (this->blockMatchingParams != NULL) {
        if (this->blockMatchingParams->totalBlock != NULL) {
            free(this->blockMatchingParams->totalBlock);
            this->blockMatchingParams->totalBlock = NULL;
        }
        if (this->blockMatchingParams->referencePosition != NULL) {
            free(this->blockMatchingParams->referencePosition);
            this->blockMatchingParams->referencePosition = NULL;
        }
        if (this->blockMatchingParams->warpedPosition != NULL) {
            free(this->blockMatchingParams->warpedPosition);
            this->blockMatchingParams->warpedPosition = NULL;
        }
        delete this->blockMatchingParams;
        this->blockMatchingParams = NULL;
    }
}
/* *************************************************************** */
bool AladinContent::isCurrentComputationDoubleCapable()
{
    return true;
}
