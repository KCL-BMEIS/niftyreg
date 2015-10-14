#include "CUDAContent.h"
#include "_reg_common_cuda.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
CudaContent::CudaContent()
{
	initVars();
	allocateCuPtrs();
	//uploadContent();
}
/* *************************************************************** */
CudaContent::CudaContent(nifti_image *CurrentReferenceIn,
								 nifti_image *CurrentFloatingIn,
								 int *CurrentReferenceMaskIn,
								 size_t byte,
								 const unsigned int blockPercentage,
								 const unsigned int inlierLts,
								 int blockStep,
								 bool cusvdIn) :
		Content(CurrentReferenceIn,
				  CurrentFloatingIn,
				  CurrentReferenceMaskIn,
				  sizeof(float), // forcing float for CUDA
				  blockPercentage,
				  inlierLts,
				  blockStep),
                  cudaSVD(cusvdIn)
{
	if(byte!=sizeof(float)){
		reg_print_fct_warn("CudaContent::CudaContent");
		reg_print_msg_warn("Datatype has been forced to float");
	}
	initVars();
	allocateCuPtrs();
	//uploadContent();

}
/* *************************************************************** */
CudaContent::CudaContent(nifti_image *CurrentReferenceIn,
								 nifti_image *CurrentFloatingIn,
								 int *CurrentReferenceMaskIn,
								 size_t byte) :
		Content(CurrentReferenceIn,
				  CurrentFloatingIn,
				  CurrentReferenceMaskIn,
				  sizeof(float)) // forcing float for CUDA
{
	if(byte!=sizeof(float)){
		reg_print_fct_warn("CudaContent::CudaContent");
		reg_print_msg_warn("Datatype has been forced to float");
	}
	initVars();
	allocateCuPtrs();
	//uploadContent();
}
/* *************************************************************** */
CudaContent::CudaContent(nifti_image *CurrentReferenceIn,
								 nifti_image *CurrentFloatingIn,
								 int *CurrentReferenceMaskIn,
								 mat44 *transMat,
								 size_t byte,
								 const unsigned int blockPercentage,
								 const unsigned int inlierLts,
								 int blockStep,
								 bool cusvdIn) :
		Content(CurrentReferenceIn,
				  CurrentFloatingIn,
				  CurrentReferenceMaskIn,
				  transMat,
				  sizeof(float), // forcing float for CUDA
				  blockPercentage,
				  inlierLts,
				  blockStep),
                  cudaSVD(cusvdIn)
{
	if(byte!=sizeof(float)){
		reg_print_fct_warn("CudaContent::CudaContent");
		reg_print_msg_warn("Datatype has been forced to float");
	}
	initVars();
	allocateCuPtrs();
	//uploadContent();
}
/* *************************************************************** */
CudaContent::CudaContent(nifti_image *CurrentReferenceIn,
								 nifti_image *CurrentFloatingIn,
								 int *CurrentReferenceMaskIn,
								 mat44 *transMat,
								 size_t byte) :
		Content(CurrentReferenceIn,
				  CurrentFloatingIn,
				  CurrentReferenceMaskIn,
				  transMat,
				  sizeof(float)) // forcing float for CUDA
{
	if(byte!=sizeof(float)){
		reg_print_fct_warn("CudaContent::CudaContent");
		reg_print_msg_warn("Datatype has been forced to float");
	}
	initVars();
	allocateCuPtrs();
	//uploadContent();
}
/* *************************************************************** */
CudaContent::~CudaContent()
{
	freeCuPtrs();
}
/* *************************************************************** */
void CudaContent::initVars()
{
    this->referenceImageArray_d = 0;
    this->floatingImageArray_d = 0;
    this->warpedImageArray_d = 0;
    this->deformationFieldArray_d = 0;
    this->referencePosition_d = 0;
    this->warpedPosition_d = 0;
    this->totalBlock_d = 0;
    this->mask_d = 0;
    this->floIJKMat_d = 0;
    this->cudaSVD = false;

	if (this->CurrentReference != NULL && this->CurrentReference->nbyper != NIFTI_TYPE_FLOAT32)
		reg_tools_changeDatatype<float>(this->CurrentReference);
	if (this->CurrentFloating != NULL && this->CurrentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
		reg_tools_changeDatatype<float>(CurrentFloating);
		if (this->CurrentWarped != NULL)
			reg_tools_changeDatatype<float>(CurrentWarped);
	}

	this->referenceVoxels = (this->CurrentReference != NULL) ? this->CurrentReference->nvox : 0;
    this->floatingVoxels = (this->CurrentFloating != NULL) ? this->CurrentFloating->nvox : 0;
    //this->numBlocks = (this->blockMatchingParams->activeBlock != NULL) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void CudaContent::allocateCuPtrs()
{

    if (this->transformationMatrix != NULL) {
        cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
        float *tmpMat_h = (float*)malloc(16 * sizeof(float));
        mat44ToCptr(*(this->transformationMatrix), tmpMat_h);

        cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
        NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
        free(tmpMat_h);
    }
    if (this->CurrentReferenceMask != NULL) {
        cudaCommon_allocateArrayToDevice<int>(&mask_d, this->referenceVoxels);
        cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels);
    }
	if (this->CurrentReference != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&referenceImageArray_d, referenceVoxels);
		cudaCommon_allocateArrayToDevice<float>(&referenceMat_d, 16);

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference);

        float* targetMat = (float *)malloc(16 * sizeof(float)); //freed
        mat44ToCptr(this->refMatrix_xyz, targetMat);
        cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, 16);
        free(targetMat);
	}
    if (this->CurrentWarped != NULL) {
        cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, this->CurrentWarped->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
    }
    if (this->CurrentDeformationField != NULL) {
        cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
        cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
    }
	if (this->CurrentFloating != NULL) {
		cudaCommon_allocateArrayToDevice<float>(&floatingImageArray_d, floatingVoxels);
		cudaCommon_allocateArrayToDevice<float>(&floIJKMat_d, 16);

        cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->CurrentFloating);

        float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
        mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
        NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
        free(sourceIJKMatrix_h);
	}

    if (this->blockMatchingParams != NULL) {
        if (this->blockMatchingParams->referencePosition != NULL) {
            cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        }
        if (this->blockMatchingParams->warpedPosition != NULL) {
            cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
            cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        }
        if (this->blockMatchingParams->totalBlock != NULL) {
            cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, blockMatchingParams->totalBlockNumber);
            cudaCommon_transferFromDeviceToNiftiSimple1<int>(&totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }

        if (this->cudaSVD) {
            /*
            unsigned int m = blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim;
            unsigned int n = 0;
            if (this->blockMatchingParams->dim == 2) {
                n = 6;
            }
            else {
                n = 12;
            }

            cudaCommon_allocateArrayToDevice<float>(&AR_d, m * n);
            cudaCommon_allocateArrayToDevice<float>(&U_d, m * m); //only the singular vectors output is needed
            cudaCommon_allocateArrayToDevice<float>(&VT_d, n * n);
            cudaCommon_allocateArrayToDevice<float>(&Sigma_d, std::min(m, n));
            cudaCommon_allocateArrayToDevice<float>(&lengths_d, blockMatchingParams->activeBlockNumber);
            cudaCommon_allocateArrayToDevice<float>(&newResultPos_d, blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
            */
        }
    }
}
/* *************************************************************** */
/*
void CudaContent::uploadContent()
{

	if (this->CurrentReferenceMask != NULL)
		cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, this->CurrentReferenceMask, referenceVoxels);

	if (this->CurrentReference != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&referenceImageArray_d, this->CurrentReference);

		float* targetMat = (float *) malloc(16 * sizeof(float)); //freed
		mat44ToCptr(this->refMatrix_xyz, targetMat);
		cudaCommon_transferFromDeviceToNiftiSimple1<float>(&referenceMat_d, targetMat, 16);
		free(targetMat);
	}

	if (this->CurrentFloating != NULL) {
		cudaCommon_transferFromDeviceToNiftiSimple<float>(&floatingImageArray_d, this->CurrentFloating);

		float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
		mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
		NR_CUDA_SAFE_CALL(cudaMemcpy(floIJKMat_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
		free(sourceIJKMatrix_h);
	}
    if (this->blockMatchingParams != NULL) {
        if (this->blockMatchingParams->totalBlock != NULL) {
            cudaCommon_transferFromDeviceToNiftiSimple1<int>(&totalBlock_d, blockMatchingParams->totalBlock, blockMatchingParams->totalBlockNumber);
        }
	}
}
*/
/* *************************************************************** */
nifti_image *CudaContent::getCurrentWarped(int type)
{
	downloadImage(CurrentWarped, warpedImageArray_d, type);
	return CurrentWarped;
}
/* *************************************************************** */
nifti_image *CudaContent::getCurrentDeformationField()
{

	cudaCommon_transferFromDeviceToCpu<float>((float*) CurrentDeformationField->data, &deformationFieldArray_d, CurrentDeformationField->nvox);
	return CurrentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* CudaContent::getBlockMatchingParams()
{

    cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->warpedPosition, &warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    cudaCommon_transferFromDeviceToCpu<float>(this->blockMatchingParams->referencePosition, &referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
	return this->blockMatchingParams;
}
/* *************************************************************** */
void CudaContent::setTransformationMatrix(mat44 *transformationMatrixIn)
{
    if (this->transformationMatrix != NULL)
        cudaCommon_free<float>(&transformationMatrix_d);

	Content::setTransformationMatrix(transformationMatrixIn);
    float *tmpMat_h = (float*)malloc(16 * sizeof(float));
    mat44ToCptr(*(this->transformationMatrix), tmpMat_h);

    cudaCommon_allocateArrayToDevice<float>(&transformationMatrix_d, 16);
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->transformationMatrix_d, tmpMat_h, 16 * sizeof(float), cudaMemcpyHostToDevice));
    free(tmpMat_h);
}
/* *************************************************************** */
void CudaContent::setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
{
	if (this->CurrentDeformationField != NULL)
		cudaCommon_free<float>(&deformationFieldArray_d);
	Content::setCurrentDeformationField(CurrentDeformationFieldIn);

	cudaCommon_allocateArrayToDevice<float>(&deformationFieldArray_d, this->CurrentDeformationField->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&deformationFieldArray_d, this->CurrentDeformationField);
}
/* *************************************************************** */
void CudaContent::setCurrentReferenceMask(int *maskIn, size_t nvox)
{
    if (this->CurrentReferenceMask != NULL)
        cudaCommon_free<int>(&mask_d);
    this->CurrentReferenceMask = maskIn;
	cudaCommon_allocateArrayToDevice<int>(&mask_d, nvox);
	cudaCommon_transferFromDeviceToNiftiSimple1<int>(&mask_d, maskIn, nvox);
}
/* *************************************************************** */
void CudaContent::setCurrentWarped(nifti_image *currentWarped)
{
	if (this->CurrentWarped != NULL)
		cudaCommon_free<float>(&warpedImageArray_d);
	Content::setCurrentWarped(currentWarped);
	reg_tools_changeDatatype<float>(this->CurrentWarped);

	cudaCommon_allocateArrayToDevice<float>(&warpedImageArray_d, CurrentWarped->nvox);
	cudaCommon_transferFromDeviceToNiftiSimple<float>(&warpedImageArray_d, this->CurrentWarped);
}
/* *************************************************************** */
void CudaContent::setBlockMatchingParams(_reg_blockMatchingParam* bmp) {

    Content::setBlockMatchingParams(bmp);
    if (this->blockMatchingParams->referencePosition != NULL) {
        cudaCommon_free<float>(&referencePosition_d);
        //referencePosition
        cudaCommon_allocateArrayToDevice<float>(&referencePosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&referencePosition_d, this->blockMatchingParams->referencePosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    }
    if (this->blockMatchingParams->warpedPosition != NULL) {
        cudaCommon_free<float>(&warpedPosition_d);
        //warpedPosition
        cudaCommon_allocateArrayToDevice<float>(&warpedPosition_d, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
        cudaCommon_transferArrayFromCpuToDevice<float>(&warpedPosition_d, this->blockMatchingParams->warpedPosition, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim);
    }
    if (this->blockMatchingParams->totalBlock != NULL) {
        cudaCommon_free<int>(&totalBlock_d);
        //activeBlock
        cudaCommon_allocateArrayToDevice<int>(&totalBlock_d, this->blockMatchingParams->totalBlockNumber);
        cudaCommon_transferArrayFromCpuToDevice<int>(&totalBlock_d, this->blockMatchingParams->totalBlock, this->blockMatchingParams->totalBlockNumber);
    }
}
/* *************************************************************** */
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
/* *************************************************************** */
template<class T>
void CudaContent::fillImageData(nifti_image *image,
										  float* memoryObject,
										  int type)
{

	size_t size = image->nvox;
	T* array = static_cast<T*>(image->data);

	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));

	if (buffer == NULL) {
		reg_print_fct_error("\nERROR: Memory allocation did not complete successfully!");
	}

	cudaCommon_transferFromDeviceToCpu<float>(buffer, &memoryObject, size);

	for (size_t i = 0; i < size; ++i) {
		array[i] = fillWarpedImageData<T>(buffer[i], type);
	}
	image->datatype = type;
	image->nbyper = sizeof(T);

	free(buffer);
}
/* *************************************************************** */
void CudaContent::downloadImage(nifti_image *image,
										  float* memoryObject,
										  int datatype)
{
	switch (datatype) {
	case NIFTI_TYPE_FLOAT32:
		fillImageData<float>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_FLOAT64:
		fillImageData<double>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_UINT8:
		fillImageData<unsigned char>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_INT8:
		fillImageData<char>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_UINT16:
		fillImageData<unsigned short>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_INT16:
		fillImageData<short>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_UINT32:
		fillImageData<unsigned int>(image, memoryObject, datatype);
		break;
	case NIFTI_TYPE_INT32:
		fillImageData<int>(image, memoryObject, datatype);
		break;
	default:
		std::cout << "CUDA: unsupported type" << std::endl;
		break;
	}
}
/* *************************************************************** */
float* CudaContent::getReferenceImageArray_d()
{
	return referenceImageArray_d;
}
/* *************************************************************** */
float* CudaContent::getFloatingImageArray_d()
{
	return floatingImageArray_d;
}
/* *************************************************************** */
float* CudaContent::getWarpedImageArray_d()
{
	return warpedImageArray_d;
}
/* *************************************************************** */
float* CudaContent::getTransformationMatrix_d()
{
	return transformationMatrix_d;
}
/* *************************************************************** */
float* CudaContent::getReferencePosition_d()
{
	return referencePosition_d;
}
/* *************************************************************** */
float* CudaContent::getWarpedPosition_d()
{
	return warpedPosition_d;
}
/* *************************************************************** */
float* CudaContent::getDeformationFieldArray_d()
{
	return deformationFieldArray_d;
}
/* *************************************************************** */
float* CudaContent::getReferenceMat_d()
{
	return referenceMat_d;
}
/* *************************************************************** */
float* CudaContent::getFloIJKMat_d()
{
	return floIJKMat_d;
}
/* *************************************************************** */
float* CudaContent::getAR_d()
{
	return AR_d;
}
/* *************************************************************** */
float* CudaContent::getU_d()
{
	return U_d;
}
/* *************************************************************** */
float* CudaContent::getVT_d()
{
	return VT_d;
}
/* *************************************************************** */
float* CudaContent::getSigma_d()
{
	return Sigma_d;
}
/* *************************************************************** */
float* CudaContent::getLengths_d()
{
	return lengths_d;
}
/* *************************************************************** */
float* CudaContent::getNewResultPos_d()
{
	return newResultPos_d;
}
/* *************************************************************** */
int *CudaContent::getTotalBlock_d()
{
	return totalBlock_d;
}
/* *************************************************************** */
int *CudaContent::getMask_d()
{
	return mask_d;
}
/* *************************************************************** */
int *CudaContent::getReferenceDims()
{
	return referenceDims;
}
/* *************************************************************** */
int *CudaContent::getFloatingDims()
{
	return floatingDims;
}
/* *************************************************************** */
void CudaContent::freeCuPtrs()
{
	if (this->transformationMatrix != NULL)
		cudaCommon_free<float>(&transformationMatrix_d);

	if (this->CurrentReference != NULL) {
		cudaCommon_free<float>(&referenceImageArray_d);
		cudaCommon_free<float>(&referenceMat_d);
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
		cudaCommon_free<int>(&totalBlock_d);
		cudaCommon_free<float>(&referencePosition_d);
		cudaCommon_free<float>(&warpedPosition_d);
		if (this->cudaSVD) {
			cudaCommon_free<float>(&AR_d);
			cudaCommon_free<float>(&U_d);
			cudaCommon_free<float>(&VT_d);
			cudaCommon_free<float>(&Sigma_d);
			cudaCommon_free<float>(&lengths_d);
			cudaCommon_free<float>(&newResultPos_d);
		}
	}
}
/* *************************************************************** */
