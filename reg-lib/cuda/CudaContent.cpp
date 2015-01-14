#include "CudaContent.h"
#include "_reg_common_gpu.h"
#include "_reg_tools.h"

CudaContent::CudaContent() {
	initVars();
	allocateCuPtrs();
	uploadContent();
}
CudaContent::CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, sizeof(float), blockPercentage, inlierLts, blockStep) {
	initVars();
	allocateCuPtrs();
	uploadContent();

}
CudaContent::CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, size_t byte) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, sizeof(float)) {
	initVars();
	allocateCuPtrs();
	uploadContent();
}

CudaContent::CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte, const unsigned int blockPercentage, const unsigned int inlierLts, int blockStep) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, transMat, sizeof(float), blockPercentage, inlierLts, blockStep) {
	initVars();
	allocateCuPtrs();
	uploadContent();

}
CudaContent::CudaContent(nifti_image* CurrentReferenceIn, nifti_image* CurrentFloatingIn, int* CurrentReferenceMaskIn, mat44* transMat, size_t byte) :
		Content(CurrentReferenceIn, CurrentFloatingIn, CurrentReferenceMaskIn, transMat, sizeof(float)) {
	initVars();
	allocateCuPtrs();
	uploadContent();
}

CudaContent::~CudaContent() {
	freeCuPtrs();

}

void CudaContent::initVars() {

	if (this->CurrentReference != NULL && this->CurrentReference->nbyper != NIFTI_TYPE_FLOAT32)
		reg_tools_changeDatatype<float>(this->CurrentReference);
	if (this->CurrentFloating != NULL && this->CurrentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
		reg_tools_changeDatatype<float>(CurrentFloating);
		if (this->CurrentWarped != NULL)
			reg_tools_changeDatatype<float>(CurrentWarped);
	}

	referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
	floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
	numBlocks = (this->blockMatchingParams != NULL) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}

void CudaContent::allocateCuPtrs() {

	if (this->transformationMatrix != NULL)
		cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
	if (this->CurrentReferenceMask != NULL)
		cudaCommon_allocateArrayToDevice<int>(&mask_d, referenceVoxels);
	if (this->CurrentReference != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceVoxels);
		cudaCommon_allocateArrayToDevice<float>(&targetMat_d, 16);
	}
	if (this->CurrentWarped != NULL)
		cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->CurrentWarped->nvox);
	if (this->CurrentDeformationField != NULL)
		cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
	if (this->CurrentFloating != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingVoxels);
		cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, 16);
	}

	if (this->blockMatchingParams != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<float>(&resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<int>(&activeBlock_d, numBlocks);
	}

}

void CudaContent::uploadContent() {

	if (this->CurrentReferenceMask != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels);

	if (this->CurrentReference != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference);

		float* targetMat = (float *) malloc(16 * sizeof(float)); //freed
		mat44ToCptr(this->refMatrix_xyz, targetMat);
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&targetMat_d, targetMat, 16);
		free(targetMat);
	}

	if (this->CurrentFloating != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->CurrentFloating);

		float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
		mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);

		//sourceIJKMatrix_d
		NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
		free(sourceIJKMatrix_h);
	}

	if (this->blockMatchingParams != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&activeBlock_d, blockMatchingParams->activeBlock, numBlocks);
	}
}

nifti_image* CudaContent::getCurrentWarped(int type) {
	downloadImage(CurrentWarped, warpedImageArray_d, true, type, "warpedImage");
	return CurrentWarped;
}

nifti_image* CudaContent::getCurrentDeformationField() {

	cudaCommon_transferFromDeviceToCpu<float>((float*) CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
	return CurrentDeformationField;
}
_reg_blockMatchingParam* CudaContent::getBlockMatchingParams() {

	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->definedActiveBlock * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->definedActiveBlock * 3);
	return blockMatchingParams;
}



void CudaContent::setTransformationMatrix(mat44* transformationMatrixIn) {
	Content::setTransformationMatrix(transformationMatrixIn);
}

void CudaContent::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn) {
	if (this->CurrentDeformationField != NULL)
		cudaCommon_free<float>(&deformationFieldArray_d);
	Content::setCurrentDeformationField(CurrentDeformationFieldIn);

	cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
}
void CudaContent::setCurrentReferenceMask(int* maskIn, size_t nvox) {

	cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
	cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}

void CudaContent::setCurrentWarped(nifti_image* currentWarped) {
	if (this->CurrentWarped != NULL)
		cudaCommon_free<float>(&warpedImageArray_d);
	Content::setCurrentWarped(currentWarped);
	reg_tools_changeDatatype<float>(this->CurrentWarped);

	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, CurrentWarped->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
}


template<class DataType>
DataType CudaContent::fillWarpedImageData(float intensity, int datatype) {

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
void CudaContent::fillImageData(nifti_image* image, float* memoryObject, bool warped, int type, std::string message) {

	size_t size = image->nvox;
	T* array = static_cast<T*>(image->data);

	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));

	if (buffer == NULL) {
		printf("\nERROR: Memory allocation did not complete successfully!");
	}

	cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

	for (size_t i = 0; i < size; ++i) {
		array[i] = fillWarpedImageData<T>(buffer[i], type);
	}
	image->datatype = type;
	image->nbyper = sizeof(T);

	free(buffer);
}

void CudaContent::downloadImage(nifti_image* image, float* memoryObject, bool flag, int datatype, std::string message) {

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
		std::cout << "CUDA: unsupported type: " << datatype << std::endl;
		break;
	}
}


float* CudaContent::getReferenceImageArray_d() {
	return referenceImageArray_d;
}
float* CudaContent::getFloatingImageArray_d() {
	return floatingImageArray_d;
}
float* CudaContent::getWarpedImageArray_d() {
	return warpedImageArray_d;
}
float* CudaContent::getTransformationMatrix_d() {
	return transformationMatrix_d;
}

float* CudaContent::getTargetPosition_d() {
	return targetPosition_d;
}
float* CudaContent::getResultPosition_d() {
	return resultPosition_d;
}
float* CudaContent::getDeformationFieldArray_d() {
	return deformationFieldArray_d;
}
float* CudaContent::getTargetMat_d() {
	return targetMat_d;
}
float* CudaContent::getFloIJKMat_d() {
	return floIJKMat_d;
}
int* CudaContent::getActiveBlock_d() {
	return activeBlock_d;
}
int* CudaContent::getMask_d() {
	return mask_d;
}

int* CudaContent::getReferenceDims() {
	return referenceDims;
}
int* CudaContent::getFloatingDims() {
	return floatingDims;
}

void CudaContent::freeCuPtrs() {

	if (this->transformationMatrix != NULL)
		cudaCommon_free<float>(&transformationMatrix_d);
	if (this->CurrentReference != NULL) {
		cudaCommon_free<float>(&referenceImageArray_d);
		cudaCommon_free<float>(&targetMat_d);
	}
	if (this->CurrentFloating != NULL) {
		cudaCommon_free<float>(&floatingImageArray_d);
		cudaCommon_free<float>(&floIJKMat_d);
	}
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
