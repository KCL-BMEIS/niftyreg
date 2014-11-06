#include "CLContext.h"
#include "_reg_common_gpu.h"


template<class T>
void ClCommon_allocateArrayToDevice(T** array, int* dims){}

template void ClCommon_allocateArrayToDevice<float>(float** devArray,  int* dims);
template void ClCommon_allocateArrayToDevice<int>(int** devArray,  int* dims);

template<class T>
void ClCommon_allocateArrayToDevice(T** array, size_t nVoxels){}

template void ClCommon_allocateArrayToDevice<float>(float** devArray,  size_t nVoxels);
template void ClCommon_allocateArrayToDevice<int>(int** devArray,  size_t nVoxels);

template<class T>
void ClCommon_transferFromDeviceToCpu( T* hostArray, T** devArray, const unsigned int nvox){}

template void ClCommon_transferFromDeviceToCpu<float>( float* data, float** devArray, const unsigned int nvox);


template<class T>
void ClCommon_transferFromDeviceToNiftiSimple(T** devArray, nifti_image* hostImage){}

template void ClCommon_transferFromDeviceToNiftiSimple<float>(float** devArray, nifti_image* hostImage);

template<class T>
void ClCommon_transferFromDeviceToNiftiSimple1(T** devArray, T* hostArray, const unsigned int nVoxels){}
template void ClCommon_transferFromDeviceToNiftiSimple1<float>(float** devArray, float* hostArray, const unsigned int nVoxels);
template void ClCommon_transferFromDeviceToNiftiSimple1<int>(int** devArray, int* hostArray, const unsigned int nVoxels);

template<class T>
void ClCommon_free(T** array){}

template void ClCommon_free<float>(float** array);
template void ClCommon_free<int>(int** array);


ClContext::~ClContext(){
	//std::cout << "Cl context destructor" << std::endl;
	freeClPtrs();
	//ClDeviceReset();
}

void ClContext::allocateClPtrs(){
	//ClDeviceReset();
	ClCommon_allocateArrayToDevice<int>(&mask_d, referenceDims);
	std::cout<<"Allocating: "<<nVoxels<<std::endl;
	ClCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceDims);
	ClCommon_allocateArrayToDevice<float>(&warpedImageArray_d, referenceDims);
	ClCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, CurrentDeformationField->nvox);
	ClCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingDims);

	if (bm){
		ClCommon_allocateArrayToDevice<float>(&targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
		ClCommon_allocateArrayToDevice<float>(&resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
		ClCommon_allocateArrayToDevice<int>(&activeBlock_d, numBlocks);
	}


}

void ClContext::initVars(){

	nifti_image* reference = getCurrentReference();
	referenceDims[1] = reference->nx;
	referenceDims[2] = reference->ny;
	referenceDims[3] = reference->nz;

	nVoxels = referenceDims[3] * referenceDims[1] * referenceDims[2];
	//std::cout << "sze1: " << nVoxels << std::endl;

	nifti_image* floating = getCurrentFloating();
	floatingDims[1] = floating->nx;
	floatingDims[2] = floating->ny;
	floatingDims[3] = floating->nz;

	if (bm) numBlocks = blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2];

}

nifti_image* ClContext::getCurrentWarped(){
	ClCommon_transferFromDeviceToCpu<float>((float*)CurrentWarped->data, &warpedImageArray_d, CurrentWarped->nvox);
	return CurrentWarped;
}

nifti_image* ClContext::getCurrentDeformationField(){
	ClCommon_transferFromDeviceToCpu<float>((float*)CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
	return CurrentDeformationField;
}
_reg_blockMatchingParam* ClContext::getBlockMatchingParams(){



	ClCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
	ClCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);

	return blockMatchingParams;
}

void ClContext::setTransformationMatrix(mat44* transformationMatrixIn){
	Context::setTransformationMatrix(transformationMatrixIn);
}

void ClContext::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn){

	Context::setCurrentDeformationField(CurrentDeformationFieldIn);
	//ClFree(deformationFieldArray_d);
	//ClCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, CurrentDeformationFieldIn->nvox);
	ClCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, Context::getCurrentDeformationField());
}

void ClContext::downloadFromClContext(){

	ClCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
	ClCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
}

void ClContext::setCurrentWarped(nifti_image* currentWarped){

	Context::setCurrentWarped(currentWarped);
	/*ClFree(warpedImageArray_d);
	ClCommon_allocateArrayToDevice<float>(&warpedImageArray_d, currentWarped->nvox);*/
	ClCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, Context::getCurrentWarped());
}
void ClContext::uploadContext(){


	ClCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, getCurrentReferenceMask(), nVoxels);
	ClCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, getCurrentReference());
	ClCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, getCurrentWarped());
	ClCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, getCurrentDeformationField());
	ClCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, getCurrentFloating());

	if (bm){
		ClCommon_transferFromDeviceToNiftiSimple1<float>(&targetPosition_d, blockMatchingParams->targetPosition, blockMatchingParams->activeBlockNumber * 3);
		ClCommon_transferFromDeviceToNiftiSimple1<float>(&resultPosition_d, blockMatchingParams->resultPosition, blockMatchingParams->activeBlockNumber * 3);
		ClCommon_transferFromDeviceToNiftiSimple1<int>(&activeBlock_d, blockMatchingParams->activeBlock, numBlocks);
	}


}
void ClContext::freeClPtrs(){

	ClCommon_free<float>(&referenceImageArray_d);
	ClCommon_free<float>(&floatingImageArray_d);
	ClCommon_free<float>(&warpedImageArray_d);
	ClCommon_free<float>(&deformationFieldArray_d);

	ClCommon_free<int>(&mask_d);
	if (bm){
		ClCommon_free<int>(&activeBlock_d);
		ClCommon_free<float>(&targetPosition_d);
		ClCommon_free<float>(&resultPosition_d);
	}


}
