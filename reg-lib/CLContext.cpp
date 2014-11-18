#include "CLContext.h"
#include "_reg_tools.h"

ClContext::~ClContext() {
	//std::cout << "Cl context destructor" << std::endl;
	freeClPtrs();

}

void ClContext::allocateClPtrs() {

	if (this->CurrentReferenceMask != NULL) {
		maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->CurrentReference->nvox * sizeof(int), this->CurrentReferenceMask, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentReferenceMask: ");
	}

	if (this->CurrentWarped != NULL) {
		warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentWarped: ");
	}

	if (this->CurrentDeformationField != NULL) {
		deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, sizeof(float) * this->CurrentDeformationField->nvox, this->CurrentDeformationField->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentDeformationField: ");
	}

	if (this->CurrentFloating != NULL) {
		floatingImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentFloating->nvox, this->CurrentFloating->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentFloating: ");
	}

	if (this->CurrentReference != NULL) {
		referenceImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentReference->nvox, this->CurrentReference->data, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentReference: ");
	}

	if (bm) {

		//targetPositionClmem
		targetPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->targetPosition, &errNum);
		sContext->checkErrNum(errNum, "failed targetPositionClmem: ");
		//resultPositionClmem
		resultPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->resultPosition, &errNum);
		sContext->checkErrNum(errNum, "failed resultPositionClmem: ");
		//activeBlockClmem
		activeBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBlocks * sizeof(int), blockMatchingParams->activeBlock, &errNum);
		sContext->checkErrNum(errNum, "failed activeBlockClmem: ");
	}
}

void ClContext::initVars() {

	referenceImageClmem=0;
	floatingImageClmem=0;
	warpedImageClmem=0;
	deformationFieldClmem=0;
	targetPositionClmem=0;
	resultPositionClmem=0;
	activeBlockClmem=0;
	maskClmem=0;

	sContext = &CLContextSingletton::Instance();
	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
//	std::cout << floatingVoxels << ": " << referenceVoxels << std::endl;
	numBlocks = (bm) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}

template<class DataType>
DataType ClContext::fillWarpedImageData(float intensity, int datatype) {
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
void ClContext::fillImageData(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message) {

	size_t size = image->nx * image->ny * image->nz;
	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));

	if (buffer == NULL) {
		printf("\nERROR: Memory allocation did not complete successfully!");
	}

	warpedImageBuffer = static_cast<float*>(this->CurrentWarped->data);
	for (int i = 0; i < 100; ++i) {
		if (warpedImageBuffer[i] > 0.00001)
			printf("data pre idx: %d | intensity %f\n", i, warpedImageBuffer[i]);
	}

	errNum = clEnqueueReadBuffer(commandQueue, memoryObject, flag, 0, size * sizeof(float), this->CurrentWarped->data, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error reading warped buffer.");
	warpedImageBuffer = static_cast<float*>(this->CurrentWarped->data);

	for (int i = 0; i < 100; ++i) {
		printf("after idx: %d | intensity %f\n", i, warpedImageBuffer[i]);
	}

	for (size_t i = 0; i < size; ++i) {
		buffer[i] = warpedImageBuffer[i];
	}
	T* dataT = static_cast<T*>(image->data);
	for (size_t i = 0; i < size; ++i) {
		dataT[i] = fillWarpedImageData<T>(buffer[i], type);
	}

	for (int i = 0; i < 100; ++i) {
		printf("buff idx: %d | intensity %f\n", i, buffer[i]);
	}

	free(buffer);
}

/*
 template void ClContext::fillImageData<float>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<double>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<unsigned char>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<char>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<unsigned short>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<short>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 template void ClContext::fillImageData<unsigned int>(nifti_image* image, size_t size, cl_mem memoryObject, cl_mem_flags flag, int type, std::string message);
 */

void ClContext::downloadImage(nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, int datatype, std::string message) {

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
		std::cout << "unsupported type" << std::endl;
		break;
	}
}

nifti_image* ClContext::getCurrentWarped(int datatype) {

	downloadImage(this->CurrentWarped, warpedImageClmem, CL_TRUE, datatype, "warpedImageClmem");
	this->CurrentWarped->datatype = datatype;
	return CurrentWarped;
}

nifti_image* ClContext::getCurrentDeformationField() {
	errNum = clEnqueueReadBuffer(this->commandQueue, deformationFieldClmem, CL_TRUE, 0, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, 0, NULL, NULL); //CLCONTEXT
	return CurrentDeformationField;
}
_reg_blockMatchingParam* ClContext::getBlockMatchingParams() {

	resultPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->resultPosition, &errNum);
	targetPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->targetPosition, &errNum);
	return blockMatchingParams;
}

void ClContext::setTransformationMatrix(mat44* transformationMatrixIn) {
	Context::setTransformationMatrix(transformationMatrixIn);
}

void ClContext::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
	if (this->CurrentDeformationField != NULL)
			clReleaseMemObject(deformationFieldClmem);

	Context::setCurrentDeformationField(CurrentDeformationFieldIn);
	deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE,  this->CurrentDeformationField->nvox*sizeof(float), this->CurrentDeformationField->data, &errNum);
	sContext->checkErrNum(errNum, "failed CurrentDeformationField: ");
}
void ClContext::setCurrentReferenceMask(int* maskIn, size_t nvox) {

	if (this->CurrentReference != NULL)
		clReleaseMemObject(maskClmem);

	this->CurrentReferenceMask = maskIn;
	maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvox * sizeof(int), maskIn, &errNum);
}

void ClContext::setCurrentWarped(nifti_image* currentWarped) {

	if (this->CurrentWarped != NULL)
		clReleaseMemObject(warpedImageClmem);
	Context::setCurrentWarped(currentWarped);
	warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &errNum);
	sContext->checkErrNum(errNum, "failed CurrentWarped: ");
}

void ClContext::freeClPtrs() {

	std::cout << "free cl ptrs" << std::endl;
	if (this->CurrentReference != NULL)
		clReleaseMemObject(referenceImageClmem);
	if (this->CurrentFloating != NULL)
		clReleaseMemObject(floatingImageClmem);
	if (this->CurrentWarped != NULL)
		clReleaseMemObject(warpedImageClmem);
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(deformationFieldClmem);

	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(maskClmem);
	if (bm) {
		clReleaseMemObject(activeBlockClmem);
		clReleaseMemObject(targetPositionClmem);
		clReleaseMemObject(resultPositionClmem);
	}

}
