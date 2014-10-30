#include "Context.h"
#include "kernels.h"
#include "Platform.h"

#include <iostream>

#define CPUX 0
#define OCLX 1
#define CUDA 2


using namespace std;

Context::Context(){
	//std::cout << "context constructor (mock)" << std::endl;
	int dim[8] = { 2, 20 , 20, 1, 1, 1, 1, 1 };

	this->CurrentFloating = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReference = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	this->CurrentReferenceMask = NULL;
	this->bm = false;

}
Context::~Context(){
	//std::cout << "Context Destructor called" << std::endl;
	ClearWarpedImage();
	ClearDeformationField();

	if (this->bm)
	delete blockMatchingParams;
}
void Context::shout() {
	//std::cout << "context listens" << std::endl;
	Platform *platform = new Platform();
	platform->shout();
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytes, const unsigned int CurrentPercentageOfBlockToUse, const unsigned int InlierLts) :CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn)
{
	//std::cout << "context constructor called" << std::endl;
	blockMatchingParams = new _reg_blockMatchingParam();
	this->AllocateWarpedImage(bytes);
	this->AllocateDeformationField(bytes);
	this->bm = true;
	//std::cout << "typeConIn: " << CurrentReference->datatype << std::endl;
	initialise_block_matching_method(CurrentReference, blockMatchingParams, CurrentPercentageOfBlockToUse, InlierLts, CurrentReferenceMask, true);
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytes) :CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn)
{
	this->bm = false;
	this->AllocateWarpedImage(bytes);
	this->AllocateDeformationField(bytes);
}


void Context::initVars(const unsigned int platformFlagIn){

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::AllocateWarpedImage(size_t bytes)
{
	if (this->CurrentReference == NULL || this->CurrentFloating == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateWarpedImage()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference and FLoating images are not defined. Exit.\n");
		reg_exit(1);
	}

	this->CurrentWarped = nifti_copy_nim_info(this->CurrentReference);
	this->CurrentWarped->dim[0] = this->CurrentWarped->ndim = this->CurrentFloating->ndim;
	this->CurrentWarped->dim[4] = this->CurrentWarped->nt = this->CurrentFloating->nt;
	this->CurrentWarped->pixdim[4] = this->CurrentWarped->dt = 1.0;
	this->CurrentWarped->nvox =
		(size_t)this->CurrentWarped->nx *
		(size_t)this->CurrentWarped->ny *
		(size_t)this->CurrentWarped->nz *
		(size_t)this->CurrentWarped->nt;
	this->CurrentWarped->datatype = this->CurrentFloating->datatype;
	this->CurrentWarped->nbyper = this->CurrentFloating->nbyper;
	this->CurrentWarped->data = (void *)calloc(this->CurrentWarped->nvox, this->CurrentWarped->nbyper);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::ClearWarpedImage()
{
	if (this->CurrentWarped != NULL)
		nifti_image_free(this->CurrentWarped);
	this->CurrentWarped = NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::AllocateDeformationField(size_t bytes)
{
	if (this->CurrentReference == NULL)
	{
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
	else this->CurrentDeformationField->dim[5] = this->CurrentDeformationField->nu = 3;
	this->CurrentDeformationField->pixdim[5] = this->CurrentDeformationField->du = 1.0;
	this->CurrentDeformationField->dim[6] = this->CurrentDeformationField->nv = 1;
	this->CurrentDeformationField->pixdim[6] = this->CurrentDeformationField->dv = 1.0;
	this->CurrentDeformationField->dim[7] = this->CurrentDeformationField->nw = 1;
	this->CurrentDeformationField->pixdim[7] = this->CurrentDeformationField->dw = 1.0;
	this->CurrentDeformationField->nvox = (size_t)this->CurrentDeformationField->nx *
		(size_t)this->CurrentDeformationField->ny *
		(size_t)this->CurrentDeformationField->nz *
		(size_t)this->CurrentDeformationField->nt *
		(size_t)this->CurrentDeformationField->nu;
	this->CurrentDeformationField->nbyper = bytes;
	if (bytes == 4)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
	else if (bytes == 8)
		this->CurrentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
	else
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Only float or double are expected for the deformation field. Exit.\n");
		reg_exit(1);
	}
	this->CurrentDeformationField->scl_slope = 1.f;
	this->CurrentDeformationField->scl_inter = 0.f;
	this->CurrentDeformationField->data = (void *)calloc(this->CurrentDeformationField->nvox, this->CurrentDeformationField->nbyper);
	return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::ClearDeformationField()
{
	if (this->CurrentDeformationField != NULL)
		nifti_image_free(this->CurrentDeformationField);
	this->CurrentDeformationField = NULL;
}



