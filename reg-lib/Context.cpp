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

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytes, const unsigned int CurrentPercentageOfBlockToUse, const unsigned int InlierLts,int stepSize_block/*, bool symmetric*/) :CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn)
{
	//std::cout << "context constructor called" << std::endl;
	blockMatchingParams = new _reg_blockMatchingParam();
	this->AllocateWarpedImage(&this->CurrentWarped, this->CurrentReference, this->CurrentFloating,  bytes);
	this->AllocateDeformationField(&this->CurrentDeformationField, this->CurrentReference, bytes);
	this->bm = true;
	//std::cout << "typeConIn: " << CurrentReference->datatype << std::endl;
	initialise_block_matching_method(CurrentReference, blockMatchingParams, CurrentPercentageOfBlockToUse, InlierLts, stepSize_block, CurrentReferenceMask, true);





}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

Context::Context(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t bytes) :CurrentReference(CurrentReferenceIn), CurrentFloating(CurrentFloatingIn), CurrentReferenceMask(CurrentReferenceMaskIn)
{
	this->bm = false;
	this->AllocateWarpedImage(&this->CurrentWarped, this->CurrentReference, this->CurrentFloating,  bytes);
	this->AllocateDeformationField(&this->CurrentDeformationField, this->CurrentReference, bytes);
}


void Context::initVars(const unsigned int platformFlagIn){

}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::AllocateWarpedImage(nifti_image** warpedIn, nifti_image* refIn, nifti_image* floatIn, size_t bytes)
{
	if (this->CurrentReference == NULL || this->CurrentFloating == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateWarpedImage()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference and FLoating images are not defined. Exit.\n");
		reg_exit(1);
	}

	(*warpedIn) = nifti_copy_nim_info(refIn);
	(*warpedIn)->dim[0] = (*warpedIn)->ndim = floatIn->ndim;
	(*warpedIn)->dim[4] = (*warpedIn)->nt = floatIn->nt;
	(*warpedIn)->pixdim[4] = (*warpedIn)->dt = 1.0;
	(*warpedIn)->nvox =
		(size_t)(*warpedIn)->nx *
		(size_t)(*warpedIn)->ny *
		(size_t)(*warpedIn)->nz *
		(size_t)(*warpedIn)->nt;
	(*warpedIn)->datatype = floatIn->datatype;
	(*warpedIn)->nbyper = floatIn->nbyper;
	(*warpedIn)->data = (void *)calloc((*warpedIn)->nvox, (*warpedIn)->nbyper);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::ClearWarpedImage()
{
	if (this->CurrentWarped != NULL)
		nifti_image_free(this->CurrentWarped);
	this->CurrentWarped = NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

void Context::AllocateDeformationField(nifti_image** defFieldIn, nifti_image* refIn, size_t bytes)
{
	if (refIn == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference image is not defined. Exit.\n");
		reg_exit(1);
	}
	//ClearDeformationField();

	((*defFieldIn)) = nifti_copy_nim_info(refIn);
	((*defFieldIn))->dim[0] = ((*defFieldIn))->ndim = 5;
	((*defFieldIn))->dim[4] = ((*defFieldIn))->nt = 1;
	((*defFieldIn))->pixdim[4] = (*defFieldIn)->dt = 1.0;
	if (refIn->nz == 1)
		((*defFieldIn))->dim[5] = (*defFieldIn)->nu = 2;
	else ((*defFieldIn))->dim[5] = (*defFieldIn)->nu = 3;
	((*defFieldIn))->pixdim[5] = (*defFieldIn)->du = 1.0;
	((*defFieldIn))->dim[6] = (*defFieldIn)->nv = 1;
	((*defFieldIn))->pixdim[6] = (*defFieldIn)->dv = 1.0;
	((*defFieldIn))->dim[7] = (*defFieldIn)->nw = 1;
	(*defFieldIn)->pixdim[7] = (*defFieldIn)->dw = 1.0;
	(*defFieldIn)->nvox = (size_t)(*defFieldIn)->nx *
		(size_t)(*defFieldIn)->ny *
		(size_t)(*defFieldIn)->nz *
		(size_t)(*defFieldIn)->nt *
		(size_t)(*defFieldIn)->nu;
	(*defFieldIn)->nbyper = bytes;
	if (bytes == 4)
		(*defFieldIn)->datatype = NIFTI_TYPE_FLOAT32;
	else if (bytes == 8)
		(*defFieldIn)->datatype = NIFTI_TYPE_FLOAT64;
	else
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Only float or double are expected for the deformation field. Exit.\n");
		reg_exit(1);
	}
	(*defFieldIn)->scl_slope = 1.f;
	(*defFieldIn)->scl_inter = 0.f;
	(*defFieldIn)->data = (void *)calloc((*defFieldIn)->nvox, (*defFieldIn)->nbyper);
	return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void Context::ClearDeformationField()
{
	if (this->CurrentDeformationField != NULL)
		nifti_image_free(this->CurrentDeformationField);
	this->CurrentDeformationField = NULL;
}



