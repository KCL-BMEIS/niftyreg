#include "CudaContext.h" 
#include "_reg_common_gpu.h"

CudaContext::~CudaContext(){
	//std::cout << "cuda context destructor" << std::endl;
	freeCuPtrs();
	//cudaDeviceReset();
}

void CudaContext::allocateCuPtrs(){
	//cudaDeviceReset();
	cudaCommon_allocateArrayToDevice<int>(&mask_d, referenceDims);
	std::cout<<"Allocating: "<<nVoxels<<std::endl;
	cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceDims);
	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, referenceDims);
	cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, CurrentDeformationField->nvox);
	cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingDims);
	
	if (bm){
		cudaCommon_allocateArrayToDevice<float>(&targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<float>(&resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_allocateArrayToDevice<int>(&activeBlock_d, numBlocks);
	}


}

void CudaContext::initVars(){

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
	//cudaFree(deformationFieldArray_d);
	//cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, CurrentDeformationFieldIn->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, Context::getCurrentDeformationField());
}

void CudaContext::downloadFromCudaContext(){

	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->targetPosition, &targetPosition_d, blockMatchingParams->activeBlockNumber * 3);
	cudaCommon_transferFromDeviceToCpu<float>(blockMatchingParams->resultPosition, &resultPosition_d, blockMatchingParams->activeBlockNumber * 3);
}

void CudaContext::setCurrentWarped(nifti_image* currentWarped){

	Context::setCurrentWarped(currentWarped);
	/*cudaFree(warpedImageArray_d);
	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, currentWarped->nvox);*/
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, Context::getCurrentWarped());
}
void CudaContext::uploadContext(){


	cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, getCurrentReferenceMask(), nVoxels);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, getCurrentReference());
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, getCurrentWarped());
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, getCurrentDeformationField());
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, getCurrentFloating());

	if (bm){
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&targetPosition_d, blockMatchingParams->targetPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&resultPosition_d, blockMatchingParams->resultPosition, blockMatchingParams->activeBlockNumber * 3);
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&activeBlock_d, blockMatchingParams->activeBlock, numBlocks);
	}


}
void CudaContext::freeCuPtrs(){

	cudaCommon_free(&referenceImageArray_d);
	cudaCommon_free(&floatingImageArray_d);
	cudaCommon_free(&warpedImageArray_d);
	cudaCommon_free(&deformationFieldArray_d);

	cudaCommon_free(&mask_d);
	if (bm){
		cudaCommon_free(&activeBlock_d);
		cudaCommon_free(&targetPosition_d);
		cudaCommon_free(&resultPosition_d);
	}


}
