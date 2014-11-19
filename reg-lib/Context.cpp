#include "Context.h"

using namespace std;

Context::Context() {
	//std::cout << "context constructor (mock)" << std::endl;
	int dim[8] = { 2, 20, 20, 1, 1, 1, 1, 1 };

	this->CurrentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReferenceMask = NULL;
	this->bm = false;

}
Context::~Context() {

	ClearWarpedImage();
	ClearDeformationField();
	if (blockMatchingParams != NULL)
		delete blockMatchingParams;

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytesIn, const unsigned int currentPercentageOfBlockToUseIn, const unsigned int inlierLtsIn, int stepSizeBlockIn/*, bool symmetric*/) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn), bytes(bytesIn),currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),inlierLts(inlierLtsIn), stepSizeBlock(stepSizeBlockIn) {

	blockMatchingParams = new _reg_blockMatchingParam();
	initVars();

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytesIn) :
		CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn), bytes(bytesIn) {

	initVars();

}

void Context::initVars() {

	if (this->CurrentFloating != NULL && this->CurrentReference != NULL)
		this->AllocateWarpedImage();
	else
		this->CurrentWarped = NULL;
	if (this->CurrentReference != NULL)
		this->AllocateDeformationField(bytes);
	else
		this->CurrentDeformationField = NULL;

	if (blockMatchingParams != NULL)
		initialise_block_matching_method(CurrentReference, blockMatchingParams, currentPercentageOfBlockToUse, inlierLts, stepSizeBlock, CurrentReferenceMask, false);
	if (this->CurrentReferenceMask == NULL && this->CurrentReference != NULL)
		this->CurrentReferenceMask = (int *) calloc(this->CurrentReference->nx * this->CurrentReference->ny * this->CurrentReference->nz, sizeof(int));
	if (this->CurrentReferenceMask == NULL && this->CurrentWarped != NULL)
			this->CurrentReferenceMask = (int *) calloc(this->CurrentWarped->nx * this->CurrentWarped->ny * this->CurrentWarped->nz, sizeof(int));
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
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::ClearWarpedImage() {
	if (this->CurrentWarped != NULL)
		nifti_image_free(this->CurrentWarped);
	this->CurrentWarped = NULL;
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
//	std::cout<<CurrentDeformationField->nvox<<std::endl;
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
//	std::cout<<CurrentDeformationField->nvox<<std::endl;
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
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::ClearDeformationField() {
	if (this->CurrentDeformationField != NULL)
		nifti_image_free(this->CurrentDeformationField);
	this->CurrentDeformationField = NULL;
}

