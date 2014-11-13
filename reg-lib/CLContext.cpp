#include "CLContext.h"

ClContext::~ClContext() {
	//std::cout << "Cl context destructor" << std::endl;
	freeClPtrs();

}

void ClContext::fillBuffers() {
//	convertDataToFloat();
}

template<class T>
void ClContext::fillBuffer(float** buffer, T* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, std::string message) {

	*buffer = NULL;
	*buffer = (float*) malloc(size * sizeof(float));

	if (*buffer == NULL) {
		printf("\nERROR: Memory allocation did not complete successfully!");
	}

	float* ptr = *buffer;
	std::cout << message <<array[size-1]<< std::endl;
	for (size_t i = 0; i < size; ++i) {
		ptr[i] = static_cast<float> (array[i]);
	}
	std::cout << "done" << std::endl;
	*memoryObject = clCreateBuffer(this->clContext, flag, size * sizeof(float), ptr, &errNum);
	sContext->checkErrNum(errNum, message);
	/*if (!keep)
	 free(*buffer);*/

}

template<class T>
void ClContext::fillImageData(float* buffer, T* array, size_t size, cl_mem memoryObject, cl_mem_flags flag, std::string message) {

	errNum = clEnqueueReadBuffer(commandQueue, memoryObject, flag, 0, size * sizeof(float), buffer, 0, NULL, NULL);
	sContext->checkErrNum(errNum, message);

	for (size_t i = 0; i < size; ++i) {
		array[i] = static_cast<T>(buffer[i]);
	}
}

/*
 template void ClContext::fillBuffer<double>(float* buffer, double* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 template void ClContext::fillBuffer<float>(float* buffer, float* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 template void ClContext::fillBuffer<int>(float* buffer, int* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 template void ClContext::fillBuffer<char>(float* buffer, char* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 template void ClContext::fillBuffer<unsigned int>(float* buffer, unsigned int* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 template void ClContext::fillBuffer<unsigned char>(float* buffer, unsigned char* array, size_t size, cl_mem* memoryObject, cl_mem_flags flag, bool keep, char* message);
 */

void ClContext::downloadImage(float* buffer, nifti_image* image, cl_mem memoryObject, cl_mem_flags flag, std::string message) {

	switch (image->datatype) {
	case NIFTI_TYPE_FLOAT32:
		fillImageData<float>(buffer, static_cast<float*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_FLOAT64:
		fillImageData<double>(buffer, static_cast<double*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_UINT8:
		fillImageData<unsigned char>(buffer, static_cast<unsigned char*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_INT8:
		fillImageData<char>(buffer, static_cast<char*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_UINT16:
		fillImageData<unsigned short>(buffer, static_cast<unsigned short*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_INT16:
		fillImageData<short>(buffer, static_cast<short*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_UINT32:
		fillImageData<unsigned int>(buffer, static_cast<unsigned int*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	case NIFTI_TYPE_INT32:
		fillImageData<int>(buffer, static_cast<int*>(image->data), image->nvox, memoryObject, flag, message);
		break;
	default:
		std::cout << "unsupported type" << std::endl;
		break;
	}
}

void ClContext::uploadImage(float** buffer, nifti_image* image, cl_mem* memoryObject, cl_mem_flags flag, bool keep, std::string message) {

	switch (image->datatype) {
	case NIFTI_TYPE_FLOAT32:
		fillBuffer<float>(buffer, static_cast<float*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_FLOAT64:
		fillBuffer<double>(buffer, static_cast<double*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_UINT8:

		std::cout << NIFTI_TYPE_UINT8 << std::endl;
		fillBuffer<unsigned char>(buffer, static_cast<unsigned char*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_INT8:
		fillBuffer<char>(buffer, static_cast<char*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_UINT16:
		fillBuffer<unsigned short>(buffer, static_cast<unsigned short*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_INT16:
		fillBuffer<short>(buffer, static_cast<short*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_UINT32:
		fillBuffer<unsigned int>(buffer, static_cast<unsigned int*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	case NIFTI_TYPE_INT32:
		fillBuffer<int>(buffer, static_cast<int*>(image->data), image->nvox, memoryObject, flag, keep, message);
		break;
	default:
		std::cout << "unsupported type" << std::endl;
		break;
	}
}

void ClContext::allocateClPtrs() {

	if (this->CurrentReferenceMask != NULL) {
		mask_d = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->CurrentFloating->nvox * sizeof(int), this->CurrentReferenceMask, &errNum);
		sContext->checkErrNum(errNum, "failed CurrentReferenceMask: ");
	}
	if (this->CurrentWarped != NULL)
		uploadImage(&warpedBuffer, this->CurrentWarped, &warpedImageClmem, CL_MEM_READ_WRITE, true, "warpedImageClmem");
	std::cout << "1" << std::endl;
	if (this->CurrentDeformationField != NULL)
		uploadImage(&deformationFieldBuffer, this->CurrentDeformationField, &deformationFieldClmem, CL_MEM_READ_WRITE, true, "deformationFieldImageClmem");
	std::cout << "2" << std::endl;
	if (this->CurrentFloating != NULL)
		uploadImage(&floatingBuffer, this->CurrentFloating, &floatingImageClmem, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, false, "floatingImageClmem");
	std::cout << "3" << std::endl;
	if (this->CurrentReference != NULL)
		uploadImage(&referenceBuffer, this->CurrentReference, &referenceImageClmem, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, false, "referenceImageClmem");
	std::cout << "4" << std::endl;

	if (bm) {

		//targetPosition_d
		targetPosition_d = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->targetPosition, &errNum);
		sContext->checkErrNum(errNum, "failed targetPosition_d: ");
		//resultPosition_d
		resultPosition_d = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, blockMatchingParams->activeBlockNumber * 3 * sizeof(float), blockMatchingParams->resultPosition, &errNum);
		sContext->checkErrNum(errNum, "failed resultPosition_d: ");
		//activeBlock_d
		activeBlock_d = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBlocks * sizeof(int), blockMatchingParams->activeBlock, &errNum);
		sContext->checkErrNum(errNum, "failed activeBlock_d: ");
	}
	std::cout << "allocated" << std::endl;
}

void ClContext::initVars() {

	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
	std::cout << floatingVoxels << ": " << referenceVoxels << std::endl;
	numBlocks = (bm) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}

nifti_image* ClContext::getCurrentWarped() {
	downloadImage(warpedBuffer, this->CurrentWarped, warpedImageClmem, CL_TRUE, "warpedImageClmem");
	return CurrentWarped;
}

nifti_image* ClContext::getCurrentDeformationField() {
	downloadImage(deformationFieldBuffer, this->CurrentDeformationField, deformationFieldClmem, CL_TRUE, "deformationFieldImageClmem");
	return CurrentDeformationField;
}
_reg_blockMatchingParam* ClContext::getBlockMatchingParams() {

	resultPosition_d = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->resultPosition, &errNum);
	targetPosition_d = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE, sizeof(float) * blockMatchingParams->activeBlockNumber * 3, blockMatchingParams->targetPosition, &errNum);
	return blockMatchingParams;
}

void ClContext::setTransformationMatrix(mat44* transformationMatrixIn) {
	Context::setTransformationMatrix(transformationMatrixIn);
}

void ClContext::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {

	Context::setCurrentDeformationField(CurrentDeformationFieldIn);
	uploadImage(&deformationFieldBuffer, this->CurrentDeformationField, &deformationFieldClmem, CL_MEM_READ_WRITE, true, "deformationFieldImageClmem");
}

void ClContext::setCurrentWarped(nifti_image* currentWarped) {

	Context::setCurrentWarped(currentWarped);
	uploadImage(&warpedBuffer, this->CurrentWarped, &warpedImageClmem, CL_MEM_READ_WRITE, true, "warpedImageClmem");
}

void ClContext::freeClPtrs() {

	 std::cout<<"1"<<std::endl;
	if (this->CurrentReference != NULL) {
		clReleaseMemObject(referenceImageClmem);
		free(referenceBuffer);
	}
	 std::cout<<"2"<<std::endl;
	if (this->CurrentFloating != NULL) {
		clReleaseMemObject(floatingImageClmem);
		free(floatingBuffer);
	} std::cout<<"3"<<std::endl;
	if (this->CurrentWarped != NULL) {
		clReleaseMemObject(warpedImageClmem);
		free(warpedBuffer);
	} std::cout<<"4"<<std::endl;
	if (this->CurrentDeformationField != NULL) {
		clReleaseMemObject(deformationFieldClmem);
		free(deformationFieldBuffer);
	}
	 std::cout<<"5"<<std::endl;
	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(mask_d);
	 std::cout<<"6"<<std::endl;
	if (bm) {
		clReleaseMemObject(activeBlock_d);
		clReleaseMemObject(targetPosition_d);
		clReleaseMemObject(resultPosition_d);
	}
	 std::cout<<"7"<<std::endl;
}
