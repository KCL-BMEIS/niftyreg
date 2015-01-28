#include "CLContent.h"
#include "_reg_tools.h"

ClContent::ClContent() {

	initVars();
	allocateClPtrs();
}
ClContent::ClContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep ) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte, blockPercentage, inlierLts, blockStep) {
	initVars();
	allocateClPtrs();
}
ClContent::ClContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, byte) {
	initVars();
	allocateClPtrs();
}

ClContent::ClContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, transMat, byte, blockPercentage, inlierLts, blockStep) {
	initVars();
	allocateClPtrs();
}
ClContent::ClContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, transMat, byte) {
	initVars();
	allocateClPtrs();
}

ClContent::~ClContent() {
	freeClPtrs();

}

void ClContent::initVars() {

	this->referenceImageClmem = 0;
	this->floatingImageClmem = 0;
	this->warpedImageClmem = 0;
	this->deformationFieldClmem = 0;
	this->targetPositionClmem = 0;
	this->resultPositionClmem = 0;
	this->activeBlockClmem = 0;
	this->maskClmem = 0;

	if (this->CurrentReference != NULL && this->CurrentReference->nbyper != NIFTI_TYPE_FLOAT32)
		reg_tools_changeDatatype<float>(this->CurrentReference);
	if (this->CurrentFloating != NULL && this->CurrentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
		reg_tools_changeDatatype<float>(this->CurrentFloating);
		if (this->CurrentWarped != NULL)
			reg_tools_changeDatatype<float>(this->CurrentWarped);
	}
	this->sContext = &CLContextSingletton::Instance();
	this->clContext = this->sContext->getContext();
	this->commandQueue = this->sContext->getCommandQueue();
	this->referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	this->floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
	this->numBlocks = (this->blockMatchingParams != NULL) ? this->blockMatchingParams->blockNumber[0] * this->blockMatchingParams->blockNumber[1] * this->blockMatchingParams->blockNumber[2] : 0;
}

void ClContent::allocateClPtrs() {

	if (this->CurrentWarped != NULL) {
		this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &this->errNum);
		this->sContext->checkErrNum(errNum, "Constructor: failed CurrentWarped: ");
	}

	if (this->CurrentDeformationField != NULL) {
		this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentDeformationField->nvox, this->CurrentDeformationField->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "Constructor: failed CurrentDeformationField: ");
	}

	if (this->CurrentFloating != NULL) {
		this->floatingImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentFloating->nvox, this->CurrentFloating->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed CurrentFloating: ");

		float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
		mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
		this->floMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), sourceIJKMatrix_h, &this->errNum);
		free(sourceIJKMatrix_h);
	}

	if (this->CurrentReference != NULL) {
		this->referenceImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentReference->nvox, this->CurrentReference->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed CurrentReference: ");

		float* targetMat = (float *) malloc(16 * sizeof(float)); //freed
		mat44ToCptr(this->refMatrix_xyz, targetMat);
		this->refMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), targetMat, &this->errNum);
		free(targetMat);
	}

	if (this->blockMatchingParams != NULL) {

		//targetPositionClmem
		this->targetPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * 3 * sizeof(float), this->blockMatchingParams->targetPosition, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed targetPositionClmem: ");
		//resultPositionClmem
		this->resultPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * 3 * sizeof(float), this->blockMatchingParams->resultPosition, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed resultPositionClmem: ");
		//activeBlockClmem
		this->activeBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->numBlocks * sizeof(int), this->blockMatchingParams->activeBlock, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed activeBlockClmem: ");
	}
	if (this->CurrentReferenceMask != NULL) {
		this->maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->CurrentReference->nx * this->CurrentReference->ny * this->CurrentReference->nz * sizeof(int), this->CurrentReferenceMask, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "failed CurrentReferenceMask: ");
	}
}

nifti_image* ClContent::getCurrentWarped(int datatype) {
//	std::cout << "get Warped1!" << std::endl;
	downloadImage(this->CurrentWarped, this->warpedImageClmem, CL_TRUE, datatype, "warpedImageClmem");
	return this->CurrentWarped;
}

nifti_image* ClContent::getCurrentDeformationField() {
	this->errNum = clEnqueueReadBuffer(this->commandQueue, this->deformationFieldClmem, CL_TRUE, 0, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, 0, NULL, NULL); //CLCONTEXT
	this->sContext->checkErrNum(errNum, "Get: failed CurrentDeformationField: ");
	return this->CurrentDeformationField;
}
_reg_blockMatchingParam* ClContent::getBlockMatchingParams() {

	this->errNum = clEnqueueReadBuffer(this->commandQueue, resultPositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * 3, this->blockMatchingParams->resultPosition, 0, NULL, NULL); //CLCONTEXT
	this->sContext->checkErrNum(this->errNum, "CLContext: failed result position: ");
	this->errNum = clEnqueueReadBuffer(this->commandQueue, targetPositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * 3, this->blockMatchingParams->targetPosition, 0, NULL, NULL); //CLCONTEXT
	this->sContext->checkErrNum(this->errNum, "CLContext: failed target position: ");
	return this->blockMatchingParams;
}

void ClContent::setTransformationMatrix(mat44* transformationMatrixIn) {
	Content::setTransformationMatrix(transformationMatrixIn);
}

void ClContent::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(this->deformationFieldClmem);

	Content::setCurrentDeformationField(CurrentDeformationFieldIn);
	this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, &this->errNum);
	this->sContext->checkErrNum(this->errNum, "Set: failed CurrentDeformationField: ");
}
void ClContent::setCurrentReferenceMask(int* maskIn, size_t nvox) {

	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(maskClmem);

	this->CurrentReferenceMask = maskIn;
	maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvox * sizeof(int), this->CurrentReferenceMask, &this->errNum);
}

void ClContent::setCurrentWarped(nifti_image* currentWarped) {
	if (this->CurrentWarped != NULL) {
		clReleaseMemObject(this->warpedImageClmem);
	}
	Content::setCurrentWarped(currentWarped);
	this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &this->errNum);
	this->sContext->checkErrNum(this->errNum, "failed CurrentWarped: ");
}

cl_mem ClContent::getReferenceImageArrayClmem() {
	return this->referenceImageClmem;
}
cl_mem ClContent::getFloatingImageArrayClmem() {
	return this->floatingImageClmem;
}
cl_mem ClContent::getWarpedImageClmem() {
	return this->warpedImageClmem;
}

cl_mem ClContent::getTargetPositionClmem() {
	return this->targetPositionClmem;
}
cl_mem ClContent::getResultPositionClmem() {
	return this->resultPositionClmem;
}
cl_mem ClContent::getDeformationFieldArrayClmem() {
	return this->deformationFieldClmem;
}
cl_mem ClContent::getActiveBlockClmem() {
	return this->activeBlockClmem;
}
cl_mem ClContent::getMaskClmem() {
	return this->maskClmem;
}
cl_mem ClContent::getRefMatClmem() {
	return this->refMatClmem;
}
cl_mem ClContent::getFloMatClmem() {
	return this->floMatClmem;
}

int* ClContent::getReferenceDims() {
	return this->referenceDims;
}
int* ClContent::getFloatingDims() {
	return this->floatingDims;
}

template<class DataType>
DataType ClContent::fillWarpedImageData(float intensity, int datatype) {
	switch (datatype) {
	case NIFTI_TYPE_FLOAT32:
		return static_cast<float>(intensity);
		break;
	case NIFTI_TYPE_FLOAT64:
		return static_cast<double>(intensity);
		break;
	case NIFTI_TYPE_UINT8:
		intensity = (intensity <= 255 ? reg_round(intensity) : 255); // 255=2^8-1
		return static_cast<unsigned char>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	case NIFTI_TYPE_UINT16:
		intensity = (intensity <= 65535 ? reg_round(intensity) : 65535); // 65535=2^16-1
		return static_cast<unsigned short>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	case NIFTI_TYPE_UINT32:
		intensity = (intensity <= 4294967295 ? reg_round(intensity) : 4294967295); // 4294967295=2^32-1
		return static_cast<unsigned int>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	default:
		return static_cast<DataType>(reg_round(intensity));
		break;
	}
}

template<class T>
void ClContent::fillImageData(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message) {

	size_t size = image->nvox;
	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));

	if (buffer == NULL) {
		reg_print_fct_error("\nERROR: Memory allocation did not complete successfully!");
	}

	this->errNum = clEnqueueReadBuffer(this->commandQueue, memoryObject, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
	this->sContext->checkErrNum(this->errNum, "Error reading warped buffer.");

	T* dataT = static_cast<T*>(image->data);
	for (size_t i = 0; i < size; ++i) {
		dataT[i] = fillWarpedImageData<T>(buffer[i], type);
	}
	image->datatype = type;
	image->nbyper = sizeof(T);
	free(buffer);
}

void ClContent::downloadImage(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int datatype, std::string message) {

	switch (datatype) {
	case NIFTI_TYPE_FLOAT32:
		fillImageData<float>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_FLOAT64:
		fillImageData<double>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_UINT8:
		fillImageData<unsigned char>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_INT8:
		fillImageData<char>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_UINT16:
		fillImageData<unsigned short>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_INT16:
		fillImageData<short>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_UINT32:
		fillImageData<unsigned int>(image, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_INT32:
		fillImageData<int>(image, memoryObject, flag, datatype, message);
		break;
	default:
		std::cout << "CL: unsupported type" << std::endl;
		break;
	}
}

void ClContent::freeClPtrs() {
	if (this->CurrentReference != NULL) {
		clReleaseMemObject(this->referenceImageClmem);
		clReleaseMemObject(this->refMatClmem);
	}
	if (this->CurrentFloating != NULL) {
		clReleaseMemObject(this->floatingImageClmem);
		clReleaseMemObject(this->floMatClmem);
	}
	if (this->CurrentWarped != NULL) {

		clReleaseMemObject(this->warpedImageClmem);
	}
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(this->deformationFieldClmem);

	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(this->maskClmem);
	if (this->blockMatchingParams != NULL) {
		clReleaseMemObject(this->activeBlockClmem);
		clReleaseMemObject(this->targetPositionClmem);
		clReleaseMemObject(this->resultPositionClmem);
	}
}
