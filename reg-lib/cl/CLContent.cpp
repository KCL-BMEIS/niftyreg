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

	referenceImageClmem = 0;
	floatingImageClmem = 0;
	warpedImageClmem = 0;
	deformationFieldClmem = 0;
	targetPositionClmem = 0;
	resultPositionClmem = 0;
	activeBlockClmem = 0;
	maskClmem = 0;

	if (this->CurrentReference != NULL && this->CurrentReference->nbyper != NIFTI_TYPE_FLOAT32)
		reg_tools_changeDatatype<float>(this->CurrentReference);
	if (this->CurrentFloating != NULL && this->CurrentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
		reg_tools_changeDatatype<float>(CurrentFloating);
		if (this->CurrentWarped != NULL)
			reg_tools_changeDatatype<float>(CurrentWarped);
	}
	sContext = &CLContextSingletton::Instance();
	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
	numBlocks = (this->blockMatchingParams != NULL) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}

void ClContent::allocateClPtrs() {

	if (this->CurrentWarped != NULL) {
		warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &errNum);
		sContext->checkErrNum(errNum, "Constructor: failed CurrentWarped: ");
	}

	if (this->CurrentDeformationField != NULL) {
		deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentDeformationField->nvox, this->CurrentDeformationField->data, &errNum);
		sContext->checkErrNum(errNum, "Constructor: failed CurrentDeformationField: ");
	}

	if (this->CurrentFloating != NULL) {
		floatingImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentFloating->nvox, this->CurrentFloating->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentFloating: ");

		float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
		mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
		floMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), sourceIJKMatrix_h, &errNum);
		free(sourceIJKMatrix_h);
	}

	if (this->CurrentReference != NULL) {
		referenceImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentReference->nvox, this->CurrentReference->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentReference: ");

		float* targetMat = (float *) malloc(16 * sizeof(float)); //freed
		mat44ToCptr(this->refMatrix_xyz, targetMat);
		refMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), targetMat, &errNum);
		free(targetMat);
	}

	if (this->blockMatchingParams != NULL) {

		//targetPositionClmem
		targetPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->targetPosition, &errNum);
		sContext->checkErrNum(errNum, "failed targetPositionClmem: ");
		//resultPositionClmem
		resultPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->resultPosition, &errNum);
		sContext->checkErrNum(errNum, "failed resultPositionClmem: ");
		//activeBlockClmem
		activeBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBlocks * sizeof(int), blockMatchingParams->activeBlock, &errNum);
		sContext->checkErrNum(errNum, "failed activeBlockClmem: ");
	}
	if (this->CurrentReferenceMask != NULL) {
		maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->CurrentReference->nx * this->CurrentReference->ny * this->CurrentReference->nz * sizeof(int), this->CurrentReferenceMask, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentReferenceMask: ");
	}
}

nifti_image* ClContent::getCurrentWarped(int datatype) {
//	std::cout << "get Warped1!" << std::endl;
	downloadImage(this->CurrentWarped, warpedImageClmem, CL_TRUE, datatype, "warpedImageClmem");
	return this->CurrentWarped;
}

nifti_image* ClContent::getCurrentDeformationField() {
	errNum = clEnqueueReadBuffer(this->commandQueue, deformationFieldClmem, CL_TRUE, 0, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, 0, NULL, NULL); //CLCONTEXT
	sContext->checkErrNum(errNum, "Get: failed CurrentDeformationField: ");
	return CurrentDeformationField;
}
_reg_blockMatchingParam* ClContent::getBlockMatchingParams() {

	errNum = clEnqueueReadBuffer(this->commandQueue, resultPositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->resultPosition, 0, NULL, NULL); //CLCONTEXT
	sContext->checkErrNum(errNum, "CLContext: failed result position: ");
	errNum = clEnqueueReadBuffer(this->commandQueue, targetPositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->targetPosition, 0, NULL, NULL); //CLCONTEXT
	sContext->checkErrNum(errNum, "CLContext: failed target position: ");
	return blockMatchingParams;
}

void ClContent::setTransformationMatrix(mat44* transformationMatrixIn) {
	Content::setTransformationMatrix(transformationMatrixIn);
}

void ClContent::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(deformationFieldClmem);

	Content::setCurrentDeformationField(CurrentDeformationFieldIn);
	deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, &errNum);
	sContext->checkErrNum(errNum, "Set: failed CurrentDeformationField: ");
}
void ClContent::setCurrentReferenceMask(int* maskIn, size_t nvox) {

	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(maskClmem);

	this->CurrentReferenceMask = maskIn;
	maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvox * sizeof(int), this->CurrentReferenceMask, &errNum);
}

void ClContent::setCurrentWarped(nifti_image* currentWarped) {
	if (this->CurrentWarped != NULL) {
		clReleaseMemObject(warpedImageClmem);
	}
	Content::setCurrentWarped(currentWarped);
	warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &errNum);
	sContext->checkErrNum(errNum, "failed CurrentWarped: ");
}

cl_mem ClContent::getReferenceImageArrayClmem() {
	return referenceImageClmem;
}
cl_mem ClContent::getFloatingImageArrayClmem() {
	return floatingImageClmem;
}
cl_mem ClContent::getWarpedImageClmem() {
	return warpedImageClmem;
}

cl_mem ClContent::getTargetPositionClmem() {
	return targetPositionClmem;
}
cl_mem ClContent::getResultPositionClmem() {
	return resultPositionClmem;
}
cl_mem ClContent::getDeformationFieldArrayClmem() {
	return deformationFieldClmem;
}
cl_mem ClContent::getActiveBlockClmem() {
	return activeBlockClmem;
}
cl_mem ClContent::getMaskClmem() {
	return maskClmem;
}
cl_mem ClContent::getRefMatClmem() {
	return refMatClmem;
}
cl_mem ClContent::getFloMatClmem() {
	return floMatClmem;
}

int* ClContent::getReferenceDims() {
	return referenceDims;
}
int* ClContent::getFloatingDims() {
	return floatingDims;
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

	errNum = clEnqueueReadBuffer(this->commandQueue, memoryObject, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error reading warped buffer.");

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
		clReleaseMemObject(referenceImageClmem);
		clReleaseMemObject(refMatClmem);
	}
	if (this->CurrentFloating != NULL) {
		clReleaseMemObject(floatingImageClmem);
		clReleaseMemObject(floMatClmem);
	}
	if (this->CurrentWarped != NULL) {

		clReleaseMemObject(warpedImageClmem);
	}
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(deformationFieldClmem);

	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(maskClmem);
	if (this->blockMatchingParams != NULL) {
		clReleaseMemObject(activeBlockClmem);
		clReleaseMemObject(targetPositionClmem);
		clReleaseMemObject(resultPositionClmem);
	}
}
