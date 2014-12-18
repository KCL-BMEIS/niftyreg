#include "Context.h"

using namespace std;

Context::Context() {
	int dim[8] = { 2, 20, 20, 1, 1, 1, 1, 1 };

	this->CurrentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReferenceMask = NULL;

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t bytesIn, const unsigned int currentPercentageOfBlockToUseIn, const unsigned int inlierLtsIn, int stepSizeBlockIn/*, bool symmetric*/) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn), transformationMatrix(transMat), bytes(bytesIn), currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn), inlierLts(inlierLtsIn), stepSizeBlock(stepSizeBlockIn) {

	blockMatchingParams = new _reg_blockMatchingParam();
	initVars();

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn,mat44* transMat, size_t bytesIn) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn),transformationMatrix(transMat),  bytes(bytesIn) {
	blockMatchingParams = NULL;
	initVars();
}


/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytesIn, const unsigned int currentPercentageOfBlockToUseIn, const unsigned int inlierLtsIn, int stepSizeBlockIn/*, bool symmetric*/) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn), bytes(bytesIn), currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn), inlierLts(inlierLtsIn), stepSizeBlock(stepSizeBlockIn) {

	blockMatchingParams = new _reg_blockMatchingParam();
	initVars();

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytesIn) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn), bytes(bytesIn) {
	blockMatchingParams = NULL;
	initVars();

}

Context::~Context() {
	ClearWarpedImage();
	ClearDeformationField();
	if (blockMatchingParams != NULL)
		delete blockMatchingParams;
}

void Context::initVars() {

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
		initialise_block_matching_method(CurrentReference, blockMatchingParams, currentPercentageOfBlockToUse, inlierLts, stepSizeBlock, CurrentReferenceMask, false);
#ifndef NDEBUG
	if(this->CurrentReference==NULL) printf("Context Warning: CurrentReference image is NULL\n");
	if(this->CurrentFloating==NULL) printf("Context Warning: CurrentFloating image is NULL\n");
	if(this->CurrentDeformationField==NULL) printf("Context Warning: CurrentDeformationField image is NULL\n");
	if(this->CurrentWarped==NULL) printf("Context Warning: CurrentWarped image is NULL\n");
	if(this->CurrentReferenceMask==NULL) printf("Context Warning: CurrentReferenceMask image is NULL\n");
	if(this->blockMatchingParams==NULL) printf("Context Warning: blockMatchingParams image is NULL\n");
#endif
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::AllocateWarpedImage() {
	if (this->CurrentReference == NULL || this->CurrentFloating == NULL) {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateWarpedImage()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference and this->CurrentFloatingg images are not defined. Exit.\n");
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

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::AllocateDeformationField(size_t bytes) {
	if (this->CurrentReference == NULL) {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference image is not defined. Exit.\n");
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
	this->CurrentDeformationField->nvox = (size_t) this->CurrentDeformationField->nx * (size_t) this->CurrentDeformationField->ny * (size_t) this->CurrentDeformationField->nz * (size_t) this->CurrentDeformationField->nt * (size_t) this->CurrentDeformationField->nu;
	this->CurrentDeformationField->nbyper = bytes;
	if (bytes == 4)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
	else if (bytes == 8)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
	else {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Only float or double are expected for the deformation field. Exit.\n");
		reg_exit(1);
	}
	this->CurrentDeformationField->scl_slope = 1.f;
	this->CurrentDeformationField->scl_inter = 0.f;
	this->CurrentDeformationField->data = (void *) calloc(this->CurrentDeformationField->nvox, this->CurrentDeformationField->nbyper);
	return;
}
void Context::setOverlapLength(const int voxelCaptureRangeIn){
		blockMatchingParams->voxelCaptureRange = voxelCaptureRangeIn;
	}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::ClearDeformationField() {
	if (this->CurrentDeformationField != NULL)
		nifti_image_free(this->CurrentDeformationField);
	this->CurrentDeformationField = NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::ClearWarpedImage() {
	if (this->CurrentWarped != NULL)
		nifti_image_free(this->CurrentWarped);
	this->CurrentWarped = NULL;
}

