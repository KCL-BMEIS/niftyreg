#include "CudaContext.h" 
#include "_reg_common_gpu.h"

CudaContext::~CudaContext() {
	//std::cout << "cuda context destructor" << std::endl;
	freeCuPtrs();
	//cudaDeviceReset();
}

void CudaContext::allocateCuPtrs() {
	//cudaDeviceReset();
	if (this->CurrentReferenceMask != NULL)
		cudaCommon_allocateArrayToDevice<int>(&mask_d, referenceVoxels);
	if (this->CurrentReference != NULL)
		cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceVoxels);
	if (this->CurrentWarped != NULL)
		cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->CurrentWarped->nvox);
	if (this->CurrentDeformationField != NULL)
		cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
	if (this->CurrentFloating != NULL)
		cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingVoxels);

	if (this->blockMatchingParams != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<float>(&resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<int>(&activeBlock_d, numBlocks);
	}

}


void CudaContext::initVars() {

	referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
	numBlocks = (this->blockMatchingParams != NULL) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
//	std::cout << referenceVoxels << ": " << floatingVoxels << " : " << numBlocks << std::endl;
}

nifti_image* CudaContext::getCurrentWarped(int type) {
	downloadImage( CurrentWarped, warpedImageArray_d, true, type,"warpedImage");
	CurrentWarped->datatype = type;
	return CurrentWarped;
}

nifti_image* CudaContext::getCurrentDeformationField() {

	cudaCommon_transferFromDeviceToCpu<float>((float*) CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
	return CurrentDeformationField;
}
_reg_blockMatchingParam* CudaContext::getBlockMatchingParams() {

	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);

	return blockMatchingParams;
}

void CudaContext::setTransformationMatrix(mat44* transformationMatrixIn) {
	Context::setTransformationMatrix(transformationMatrixIn);
}

void CudaContext::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
	if (this->CurrentDeformationField != NULL)
		cudaCommon_free<float>(&deformationFieldArray_d);
	Context::setCurrentDeformationField(CurrentDeformationFieldIn);

	cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
}
void CudaContext::setCurrentReferenceMask(int* maskIn, size_t nvox) {

	cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
	cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}

void CudaContext::setCurrentWarped(nifti_image* currentWarped) {
	if (this->CurrentWarped != NULL)
		cudaCommon_free<float>(&warpedImageArray_d);
	Context::setCurrentWarped(currentWarped);

	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, CurrentWarped->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
}
void CudaContext::downloadFromCudaContext() {

	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
}
void CudaContext::uploadContext() {

	if (this->CurrentDeformationField != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);

	if (this->CurrentReferenceMask != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels);

	if (this->CurrentReference != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference);

	if (this->CurrentWarped != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);

	if (this->CurrentFloating != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->CurrentFloating);

	if (this->blockMatchingParams != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&targetPosition_d, blockMatchingParams->targetPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&resultPosition_d, blockMatchingParams->resultPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&activeBlock_d, blockMatchingParams->activeBlock, numBlocks);
	}
}
template<class DataType>
DataType CudaContext::fillWarpedImageData(float intensity, int datatype) {

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
void CudaContext::fillImageData( T* array, size_t size, float* memoryObject, bool warped, int type, std::string message) {

	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));

	if (buffer == NULL) {
		printf("\nERROR: Memory allocation did not complete successfully!");
	}

	cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

	for (size_t i = 0; i < size; ++i) {
		array[i] = fillWarpedImageData<T>(buffer[i], type);
	}

	free(buffer);
}

void CudaContext::downloadImage(  nifti_image* image, float* memoryObject, bool flag, int datatype, std::string message) {

	switch (datatype) {
	case NIFTI_TYPE_FLOAT32:
		fillImageData<float>( static_cast<float*>(image->data), image->nvox, memoryObject, flag, datatype, message);
		break;
	case NIFTI_TYPE_FLOAT64:
		fillImageData<double>( static_cast<double*>(image->data), image->nvox, memoryObject, flag, datatype,message);
		break;
	case NIFTI_TYPE_UINT8:
		fillImageData<unsigned char>( static_cast<unsigned char*>(image->data), image->nvox, memoryObject, flag, datatype,message);
		break;
	case NIFTI_TYPE_INT8:
		fillImageData<char>( static_cast<char*>(image->data), image->nvox, memoryObject, flag, datatype,message);
		break;
	case NIFTI_TYPE_UINT16:
		fillImageData<unsigned short>( static_cast<unsigned short*>(image->data), image->nvox, memoryObject, flag, datatype,message);
		break;
	case NIFTI_TYPE_INT16:
		fillImageData<short>( static_cast<short*>(image->data), image->nvox, memoryObject, flag, datatype,message);
		break;
	case NIFTI_TYPE_UINT32:
		fillImageData<unsigned int>( static_cast<unsigned int*>(image->data), image->nvox, memoryObject, flag,datatype, message);
		break;
	case NIFTI_TYPE_INT32:
		fillImageData<int>( static_cast<int*>(image->data), image->nvox, memoryObject, flag,datatype, message);
		break;
	default:
		std::cout << "CUDA: unsupported type: "<< datatype<< std::endl;
		break;
	}
}

void CudaContext::freeCuPtrs() {

	if (this->CurrentReference != NULL)
		cudaCommon_free<float>(&referenceImageArray_d);
	if (this->CurrentFloating != NULL)
		cudaCommon_free<float>(&floatingImageArray_d);
	if (this->CurrentWarped != NULL)
		cudaCommon_free<float>(&warpedImageArray_d);
	if (this->CurrentDeformationField != NULL)
		cudaCommon_free<float>(&deformationFieldArray_d);

	if (this->CurrentReferenceMask != NULL)
		cudaCommon_free<int>(&mask_d);
	if (this->blockMatchingParams != NULL) {
		cudaCommon_free<int>(&activeBlock_d);
		cudaCommon_free<float>(&targetPosition_d);
		cudaCommon_free<float>(&resultPosition_d);
	}

}
