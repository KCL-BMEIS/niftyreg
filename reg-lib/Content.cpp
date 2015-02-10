#include "Content.h"

using namespace std;

/* *************************************************************** */
Content::Content()
{
	int dim[8] = { 2, 20, 20, 1, 1, 1, 1, 1 };
	this->CurrentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReferenceMask = NULL;
	initVars();
}
/* *************************************************************** */
Content::Content(nifti_image *CurrentReferenceIn,
					  nifti_image *CurrentFloatingIn,
					  int *CurrentReferenceMaskIn,
					  mat44 *transMat,
					  size_t bytesIn,
					  const unsigned int currentPercentageOfBlockToUseIn,
					  const unsigned int inlierLtsIn,
					  int stepSizeBlockIn) :
		CurrentReference(CurrentReferenceIn),
		CurrentFloating(CurrentFloatingIn),
		CurrentReferenceMask(CurrentReferenceMaskIn),
		transformationMatrix(transMat),
		bytes(bytesIn),
		currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
		inlierLts(inlierLtsIn),
		stepSizeBlock(stepSizeBlockIn)
{
	this->blockMatchingParams = new _reg_blockMatchingParam();
	initVars();
}
/* *************************************************************** */
Content::Content(nifti_image *CurrentReferenceIn,
					  nifti_image *CurrentFloatingIn,
					  int *CurrentReferenceMaskIn,
					  mat44 *transMat,
					  size_t bytesIn) :
		CurrentReference(CurrentReferenceIn),
		CurrentFloating(CurrentFloatingIn),
		CurrentReferenceMask(CurrentReferenceMaskIn),
		transformationMatrix(transMat),
		bytes(bytesIn)
{
	this->blockMatchingParams = NULL;
	initVars();
}
/* *************************************************************** */
Content::Content(nifti_image *CurrentReferenceIn,
					  nifti_image *CurrentFloatingIn,
					  int *CurrentReferenceMaskIn,
					  size_t bytesIn,
					  const unsigned int currentPercentageOfBlockToUseIn,
					  const unsigned int inlierLtsIn,
					  int stepSizeBlockIn) :
		CurrentReference(CurrentReferenceIn),
		CurrentFloating(CurrentFloatingIn),
		CurrentReferenceMask(CurrentReferenceMaskIn),
		bytes(bytesIn),
		currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
		inlierLts(inlierLtsIn),
		stepSizeBlock(stepSizeBlockIn)
{
	this->blockMatchingParams = new _reg_blockMatchingParam();
	initVars();
}
/* *************************************************************** */
Content::Content(nifti_image *CurrentReferenceIn,
					  nifti_image *CurrentFloatingIn,
					  int *CurrentReferenceMaskIn,
					  size_t bytesIn) :
		CurrentReference(CurrentReferenceIn),
		CurrentFloating(CurrentFloatingIn),
		CurrentReferenceMask(CurrentReferenceMaskIn),
		bytes(bytesIn)
{
	this->blockMatchingParams = NULL;
	initVars();
}
/* *************************************************************** */
Content::~Content()
{
	ClearWarpedImage();
	ClearDeformationField();
	if (this->blockMatchingParams != NULL)
		delete this->blockMatchingParams;
}
/* *************************************************************** */
void Content::initVars()
{
	if (this->CurrentFloating != NULL && this->CurrentReference != NULL)
		this->AllocateWarpedImage();
	else
		this->CurrentWarped = NULL;

	if (this->CurrentReference != NULL){
		this->AllocateDeformationField(bytes);
		refMatrix_xyz = (CurrentReference->sform_code > 0) ? (CurrentReference->sto_xyz) : (CurrentReference->qto_xyz);
	}
	else
		this->CurrentDeformationField = NULL;

	if (this->CurrentReferenceMask == NULL && this->CurrentReference != NULL)
		this->CurrentReferenceMask = (int *) calloc(this->CurrentReference->nx * this->CurrentReference->ny * this->CurrentReference->nz, sizeof(int));


	if (this->CurrentFloating != NULL){
		floMatrix_ijk = (CurrentFloating->sform_code > 0) ? (CurrentFloating->sto_ijk) :  (CurrentFloating->qto_ijk);
	}
	if (blockMatchingParams != NULL)
		initialise_block_matching_method(CurrentReference,
													blockMatchingParams,
													currentPercentageOfBlockToUse,
													inlierLts,
													stepSizeBlock,
													CurrentReferenceMask,
													false);
#ifndef NDEBUG
	if(this->CurrentReference==NULL) reg_print_msg_debug("CurrentReference image is NULL");
	if(this->CurrentFloating==NULL) reg_print_msg_debug("CurrentFloating image is NULL");
	if(this->CurrentDeformationField==NULL) reg_print_msg_debug("CurrentDeformationField image is NULL");
	if(this->CurrentWarped==NULL) reg_print_msg_debug("CurrentWarped image is NULL");
	if(this->CurrentReferenceMask==NULL) reg_print_msg_debug("CurrentReferenceMask image is NULL");
	if(this->blockMatchingParams==NULL) reg_print_msg_debug("blockMatchingParams image is NULL");
#endif
}
/* *************************************************************** */
void Content::AllocateWarpedImage()
{
	if (this->CurrentReference == NULL || this->CurrentFloating == NULL) {
		reg_print_fct_error( "Content::AllocateWarpedImage()");
		reg_print_msg_error(" Reference and floating images are not defined. Exit.");
		reg_exit(1);
	}

	this->CurrentWarped = nifti_copy_nim_info(this->CurrentReference);
	this->CurrentWarped->dim[0] = this->CurrentWarped->ndim = this->CurrentFloating->ndim;
	this->CurrentWarped->dim[4] = this->CurrentWarped->nt = this->CurrentFloating->nt;
	this->CurrentWarped->pixdim[4] = this->CurrentWarped->dt = 1.0;
	this->CurrentWarped->nvox = (size_t) this->CurrentWarped->nx * (size_t) this->CurrentWarped->ny * (size_t) this->CurrentWarped->nz * (size_t) this->CurrentWarped->nt;
	this->CurrentWarped->datatype = this->CurrentFloating->datatype;
	this->CurrentWarped->nbyper = this->CurrentFloating->nbyper;
	this->CurrentWarped->data = (void *) calloc(this->CurrentWarped->nvox, this->CurrentWarped->nbyper);
	this->floatingDatatype = this->CurrentFloating->datatype;
}
/* *************************************************************** */
void Content::AllocateDeformationField(size_t bytes)
{
	if (this->CurrentReference == NULL) {
		reg_print_fct_error( "Content::AllocateDeformationField()");
		reg_print_msg_error("Reference image is not defined. Exit.");
		reg_exit(1);
	}
	//ClearDeformationField();

	this->CurrentDeformationField = nifti_copy_nim_info(this->CurrentReference);
	this->CurrentDeformationField->dim[0] = this->CurrentDeformationField->ndim = 5;
	this->CurrentDeformationField->dim[4] = this->CurrentDeformationField->nt = 1;
	this->CurrentDeformationField->pixdim[4] = this->CurrentDeformationField->dt = 1.0;
	if (this->CurrentReference->nz == 1)
		this->CurrentDeformationField->dim[5] = this->CurrentDeformationField->nu = 2;
	else
		this->CurrentDeformationField->dim[5] = this->CurrentDeformationField->nu = 3;
	this->CurrentDeformationField->pixdim[5] = this->CurrentDeformationField->du = 1.0;
	this->CurrentDeformationField->dim[6] = this->CurrentDeformationField->nv = 1;
	this->CurrentDeformationField->pixdim[6] = this->CurrentDeformationField->dv = 1.0;
	this->CurrentDeformationField->dim[7] = this->CurrentDeformationField->nw = 1;
	this->CurrentDeformationField->pixdim[7] = this->CurrentDeformationField->dw = 1.0;
	this->CurrentDeformationField->nvox = (size_t) this->CurrentDeformationField->nx *
			this->CurrentDeformationField->ny * this->CurrentDeformationField->nz *
			this->CurrentDeformationField->nt * this->CurrentDeformationField->nu;
	this->CurrentDeformationField->nbyper = bytes;
	if (bytes == 4)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
	else if (bytes == 8)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
	else {
		reg_print_fct_error( "Content::AllocateDeformationField()");
		reg_print_msg_error( "Only float or double are expected for the deformation field. Exit.");
		reg_exit(1);
	}
	this->CurrentDeformationField->scl_slope = 1.f;
	this->CurrentDeformationField->scl_inter = 0.f;
	this->CurrentDeformationField->data = (void *) calloc(this->CurrentDeformationField->nvox, this->CurrentDeformationField->nbyper);
	return;
}
/* *************************************************************** */
void Content::setCaptureRange(const int voxelCaptureRangeIn)
{
	this->blockMatchingParams->voxelCaptureRange = voxelCaptureRangeIn;
}
/* *************************************************************** */
void Content::ClearDeformationField()
{
	if (this->CurrentDeformationField != NULL)
		nifti_image_free(this->CurrentDeformationField);
	this->CurrentDeformationField = NULL;
}
/* *************************************************************** */
void Content::ClearWarpedImage()
{
	if (this->CurrentWarped != NULL)
		nifti_image_free(this->CurrentWarped);
	this->CurrentWarped = NULL;
}
/* *************************************************************** */

