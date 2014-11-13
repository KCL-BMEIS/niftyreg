#include "CudaContext.h" 
#include "_reg_common_gpu.h"

CudaContext::~CudaContext(){
	//std::cout << "cuda context destructor" << std::endl;
	freeCuPtrs();
	//cudaDeviceReset();
}

void CudaContext::allocateCuPtrs(){
	//cudaDeviceReset();
	cudaCommon_allocateArrayToDevice<int>(&mask_d, referenceVoxels);
	cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceVoxels);
	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, referenceVoxels);
	cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, referenceVoxels);
	cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingVoxels);
	
	if (bm){
		cudaCommon_allocateArrayToDevice<float>(&targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<float>(&resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<int>(&activeBlock_d, numBlocks);
	}


}

void CudaContext::initVars(){

	referenceVoxels = (this->CurrentReference!= NULL)?this->CurrentReference->nvox:0;
	floatingVoxels = (this->CurrentFloating!= NULL)?this->CurrentFloating->nvox:0;
	numBlocks = (bm)?blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2]:0;
	std::cout<<referenceVoxels<<": "<<floatingVoxels<<":"<<numBlocks<<std::endl;
}

nifti_image* CudaContext::getCurrentWarped(){
	cudaCommon_transferFromDeviceToCpu<float>((float*)CurrentWarped->data, &warpedImageArray_d, CurrentWarped->nvox);
	return CurrentWarped;
}

nifti_image* CudaContext::getCurrentDeformationField(){
	cudaCommon_transferFromDeviceToCpu<float>((float*)CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
	return CurrentDeformationField;
}
_reg_blockMatchingParam* CudaContext::getBlockMatchingParams(){


	
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);

	return blockMatchingParams;
}

void CudaContext::setTransformationMatrix(mat44* transformationMatrixIn){
	Context::setTransformationMatrix(transformationMatrixIn);
}

void CudaContext::setCurrentDeformationField(nifti_image* CurrentDeformationFieldIn){

	Context::setCurrentDeformationField(CurrentDeformationFieldIn);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, Context::getCurrentDeformationField());
}

void CudaContext::downloadFromCudaContext(){

	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
}

void CudaContext::setCurrentWarped(nifti_image* currentWarped){

	Context::setCurrentWarped(currentWarped);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, Context::getCurrentWarped());
}
void CudaContext::uploadContext(){

	nifti_image* def = Context::getCurrentDeformationField();
	 std::cout<<"refs: "<<this->CurrentReference->nvox<<":"<<referenceVoxels<<":"<<this->CurrentDeformationField->nvox<<": "<<def->nvox<<std::endl;
	 if(this->CurrentDeformationField!= NULL) cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField); std::cout<<"1"<<std::endl;
	if(this->CurrentReferenceMask != NULL) cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels); std::cout<<"2"<<std::endl;
	if(this->CurrentReference != NULL)cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference); std::cout<<"3"<<std::endl;
	if(this->CurrentWarped != NULL) cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, getCurrentWarped()); std::cout<<"4"<<std::endl;

	if(this->CurrentFloating!= NULL) cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, getCurrentFloating()); std::cout<<"1"<<std::endl;
	 std::cout<<"1"<<std::endl;
	if (bm){
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&targetPosition_d, blockMatchingParams->targetPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&resultPosition_d, blockMatchingParams->resultPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&activeBlock_d, blockMatchingParams->activeBlock, numBlocks);
	}


}
void CudaContext::freeCuPtrs(){

	if(this->CurrentReference != NULL) cudaCommon_free<float>(&referenceImageArray_d);
	if(this->CurrentFloating!= NULL) cudaCommon_free<float>(&floatingImageArray_d);
	if(this->CurrentWarped != NULL) cudaCommon_free<float>(&warpedImageArray_d);
	if(this->CurrentDeformationField!= NULL) cudaCommon_free<float>(&deformationFieldArray_d);

	if(this->CurrentReferenceMask != NULL)  cudaCommon_free<int>(&mask_d);
	if (bm){
		cudaCommon_free<int>(&activeBlock_d);
		cudaCommon_free<float>(&targetPosition_d);
		cudaCommon_free<float>(&resultPosition_d);
	}


}
