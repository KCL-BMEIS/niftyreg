#include "CLAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent()
{
	initVars();
	allocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *CurrentReferenceIn,
											nifti_image *CurrentFloatingIn,
											int *CurrentReferenceMaskIn,
											size_t byte,
											const unsigned int blockPercentage,
											const unsigned int inlierLts,
											int blockStep ) :
	AladinContent(CurrentReferenceIn,
					  CurrentFloatingIn,
					  CurrentReferenceMaskIn,
					  byte, blockPercentage,
					  inlierLts,
					  blockStep)
{
	initVars();
	allocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *CurrentReferenceIn,
											nifti_image *CurrentFloatingIn,
											int *CurrentReferenceMaskIn,
											size_t byte) :
	AladinContent(CurrentReferenceIn,
					  CurrentFloatingIn,
					  CurrentReferenceMaskIn,
					  byte)
{
	initVars();
	allocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *CurrentReferenceIn,
											nifti_image *CurrentFloatingIn,
											int *CurrentReferenceMaskIn,
											mat44 *transMat,
											size_t byte,
											const unsigned int blockPercentage,
											const unsigned int inlierLts,
											int blockStep) :
	AladinContent(CurrentReferenceIn,
					  CurrentFloatingIn,
					  CurrentReferenceMaskIn,
					  transMat,
					  byte,
					  blockPercentage,
					  inlierLts,
					  blockStep)
{
	initVars();
	allocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *CurrentReferenceIn,
											nifti_image *CurrentFloatingIn,
											int *CurrentReferenceMaskIn,
											mat44 *transMat,
											size_t byte) :
	AladinContent(CurrentReferenceIn,
					  CurrentFloatingIn,
					  CurrentReferenceMaskIn,
					  transMat,
					  byte)
{
	initVars();
	allocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::~ClAladinContent()
{
	freeClPtrs();
}
/* *************************************************************** */
void ClAladinContent::initVars()
{

	this->referenceImageClmem = 0;
	this->floatingImageClmem = 0;
	this->warpedImageClmem = 0;
	this->deformationFieldClmem = 0;
	this->referencePositionClmem = 0;
	this->warpedPositionClmem = 0;
	this->totalBlockClmem = 0;
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
	//this->numBlocks = (this->blockMatchingParams != NULL) ? this->blockMatchingParams->blockNumber[0] * this->blockMatchingParams->blockNumber[1] * this->blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void ClAladinContent::allocateClPtrs()
{

	if (this->CurrentWarped != NULL)
	{
		this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClAladinContent::allocateClPtrs failed to allocate memory (warpedImageClmem): ");
	}
	if (this->CurrentDeformationField != NULL)
	{
		this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentDeformationField->nvox, this->CurrentDeformationField->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClAladinContent::allocateClPtrs failed to allocate memory (deformationFieldClmem): ");
	}
	if (this->CurrentFloating != NULL)
	{
		this->floatingImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->CurrentFloating->nvox, this->CurrentFloating->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClAladinContent::allocateClPtrs failed to allocate memory (CurrentFloating): ");

		float *sourceIJKMatrix_h = (float*) malloc(16 * sizeof(float));
		mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
		this->floMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
													  16 * sizeof(float),
													  sourceIJKMatrix_h, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (floMatClmem): ");
		free(sourceIJKMatrix_h);
	}
	if (this->CurrentReference != NULL)
	{
		this->referenceImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
																 sizeof(float) * this->CurrentReference->nvox,
																 this->CurrentReference->data, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (referenceImageClmem): ");

		float* targetMat = (float *) malloc(16 * sizeof(float)); //freed
		mat44ToCptr(this->refMatrix_xyz, targetMat);
		this->refMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
													  16 * sizeof(float),
													  targetMat, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (refMatClmem): ");
		free(targetMat);
	}
	if (this->blockMatchingParams != NULL) {
		if (this->blockMatchingParams->referencePosition != NULL) {
			//targetPositionClmem
			this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
																		 this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
																		 this->blockMatchingParams->referencePosition, &this->errNum);
			this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (referencePositionClmem): ");
		}
		if (this->blockMatchingParams->warpedPosition != NULL) {
			//resultPositionClmem
			this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
																	 this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
																	 this->blockMatchingParams->warpedPosition, &this->errNum);
			this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (warpedPositionClmem): ");
		}
		if (this->blockMatchingParams->totalBlock != NULL) {
			//totalBlockClmem
			this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
																this->blockMatchingParams->totalBlockNumber * sizeof(int),
																this->blockMatchingParams->totalBlock, &this->errNum);
			this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (activeBlockClmem): ");
		}
	}
	if (this->CurrentReferenceMask != NULL && this->CurrentReference != NULL) {
		this->maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
													this->CurrentReference->nx * this->CurrentReference->ny * this->CurrentReference->nz * sizeof(int),
													this->CurrentReferenceMask, &this->errNum);
		this->sContext->checkErrNum(this->errNum, "ClContent::allocateClPtrs failed to allocate memory (clCreateBuffer): ");
	}
}
/* *************************************************************** */
nifti_image *ClAladinContent::getCurrentWarped(int datatype)
{
	downloadImage(this->CurrentWarped, this->warpedImageClmem, datatype);
	return this->CurrentWarped;
}
/* *************************************************************** */
nifti_image *ClAladinContent::getCurrentDeformationField()
{
	this->errNum = clEnqueueReadBuffer(this->commandQueue, this->deformationFieldClmem, CL_TRUE, 0, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, 0, NULL, NULL); //CLCONTEXT
	this->sContext->checkErrNum(errNum, "Get: failed CurrentDeformationField: ");
	return this->CurrentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* ClAladinContent::getBlockMatchingParams()
{
   this->errNum = clEnqueueReadBuffer(this->commandQueue, this->warpedPositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->warpedPosition, 0, NULL, NULL); //CLCONTEXT
   this->sContext->checkErrNum(this->errNum, "CLContext: failed result position: ");
   this->errNum = clEnqueueReadBuffer(this->commandQueue, this->referencePositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->referencePosition, 0, NULL, NULL); //CLCONTEXT
   this->sContext->checkErrNum(this->errNum, "CLContext: failed target position: ");
   return this->blockMatchingParams;
}
/* *************************************************************** */
void ClAladinContent::setTransformationMatrix(mat44 *transformationMatrixIn)
{
   AladinContent::setTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::setCurrentDeformationField(nifti_image *CurrentDeformationFieldIn)
{
	if (this->CurrentDeformationField != NULL)
		clReleaseMemObject(this->deformationFieldClmem);

	AladinContent::setCurrentDeformationField(CurrentDeformationFieldIn);
	this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->CurrentDeformationField->nvox * sizeof(float), this->CurrentDeformationField->data, &this->errNum);
	this->sContext->checkErrNum(this->errNum, "ClAladinContent::setCurrentDeformationField failed to allocate memory (deformationFieldClmem): ");
}
/* *************************************************************** */
void ClAladinContent::setCurrentReferenceMask(int *maskIn, size_t nvox)
{
	if (this->CurrentReferenceMask != NULL)
		clReleaseMemObject(maskClmem);
	this->CurrentReferenceMask = maskIn;
	this->maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvox * sizeof(int), this->CurrentReferenceMask, &this->errNum);
	this->sContext->checkErrNum(this->errNum, "ClAladinContent::setCurrentReferenceMask failed to allocate memory (maskClmem): ");
}
/* *************************************************************** */
void ClAladinContent::setCurrentWarped(nifti_image *currentWarped)
{
	if (this->CurrentWarped != NULL) {
		clReleaseMemObject(this->warpedImageClmem);
	}
	if (currentWarped->nbyper != NIFTI_TYPE_FLOAT32) {
		reg_tools_changeDatatype<float>(currentWarped);
	}
	AladinContent::setCurrentWarped(currentWarped);
	this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->CurrentWarped->nvox * sizeof(float), this->CurrentWarped->data, &this->errNum);
	this->sContext->checkErrNum(this->errNum, "ClAladinContent::setCurrentWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClAladinContent::setBlockMatchingParams(_reg_blockMatchingParam* bmp) {

   AladinContent::setBlockMatchingParams(bmp);
   if (this->blockMatchingParams->referencePosition != NULL) {
      clReleaseMemObject(this->referencePositionClmem);
      //referencePositionClmem
      this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->referencePosition, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
   }
   if (this->blockMatchingParams->warpedPosition != NULL) {
      clReleaseMemObject(this->warpedPositionClmem);
      //warpedPositionClmem
      this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->warpedPosition, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
   }
   if (this->blockMatchingParams->totalBlock != NULL) {
      clReleaseMemObject(this->totalBlockClmem);
      //totalBlockClmem
      this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->totalBlockNumber * sizeof(int), this->blockMatchingParams->totalBlock, &this->errNum);
      this->sContext->checkErrNum(this->errNum, "ClAladinContent::setBlockMatchingParams failed to allocate memory (activeBlockClmem): ");
   }
}
/* *************************************************************** */
cl_mem ClAladinContent::getReferenceImageArrayClmem()
{
   return this->referenceImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getFloatingImageArrayClmem()
{
   return this->floatingImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getWarpedImageClmem()
{
   return this->warpedImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getReferencePositionClmem()
{
   return this->referencePositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getWarpedPositionClmem()
{
   return this->warpedPositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getDeformationFieldArrayClmem()
{
   return this->deformationFieldClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getTotalBlockClmem()
{
   return this->totalBlockClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getMaskClmem()
{
   return this->maskClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getRefMatClmem()
{
   return this->refMatClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::getFloMatClmem()
{
   return this->floMatClmem;
}
/* *************************************************************** */
int *ClAladinContent::getReferenceDims()
{
	return this->referenceDims;
}
/* *************************************************************** */
int *ClAladinContent::getFloatingDims() {
	return this->floatingDims;
}
/* *************************************************************** */
template<class DataType>
DataType ClAladinContent::fillWarpedImageData(float intensity, int datatype)
{
	switch (datatype) {
	case NIFTI_TYPE_FLOAT32:
		return static_cast<float>(intensity);
		break;
	case NIFTI_TYPE_FLOAT64:
		return static_cast<double>(intensity);
		break;
	case NIFTI_TYPE_UINT8:
		if(intensity!=intensity)
			intensity=0;
		intensity = (intensity <= 255 ? reg_round(intensity) : 255); // 255=2^8-1
		return static_cast<unsigned char>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	case NIFTI_TYPE_UINT16:
		if(intensity!=intensity)
			intensity=0;
		intensity = (intensity <= 65535 ? reg_round(intensity) : 65535); // 65535=2^16-1
		return static_cast<unsigned short>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	case NIFTI_TYPE_UINT32:
		if(intensity!=intensity)
			intensity=0;
		intensity = (intensity <= 4294967295 ? reg_round(intensity) : 4294967295); // 4294967295=2^32-1
		return static_cast<unsigned int>(intensity > 0 ? reg_round(intensity) : 0);
		break;
	default:
		if(intensity!=intensity)
			intensity=0;
		return static_cast<DataType>(reg_round(intensity));
		break;
	}
}
/* *************************************************************** */
template<class T>
void ClAladinContent::fillImageData(nifti_image *image,
												cl_mem memoryObject,
												int type)
{
	size_t size = image->nvox;
	float* buffer = NULL;
	buffer = (float*) malloc(size * sizeof(float));
	if (buffer == NULL) {
		reg_print_fct_error("ClAladinContent::fillImageData");
		reg_print_msg_error("Memory allocation did not complete successfully. Exit.");
		reg_exit();
	}

	this->errNum = clEnqueueReadBuffer(this->commandQueue, memoryObject, CL_TRUE, 0,
												  size * sizeof(float), buffer, 0, NULL, NULL);
	this->sContext->checkErrNum(this->errNum, "Error reading warped buffer.");

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = (void *)malloc(image->nvox*image->nbyper);
    T* dataT = static_cast<T*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = fillWarpedImageData<T>(buffer[i], type);
    free(buffer);
}
/* *************************************************************** */
void ClAladinContent::downloadImage(nifti_image *image,
												cl_mem memoryObject,
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
		reg_print_fct_error("ClAladinContent::downloadImage");
		reg_print_msg_error("Unsupported type");
		reg_exit();
		break;
	}
}
/* *************************************************************** */
void ClAladinContent::freeClPtrs()
{
	if(this->CurrentReference != NULL)
	{
		clReleaseMemObject(this->referenceImageClmem);
		clReleaseMemObject(this->refMatClmem);
	}
	if(this->CurrentFloating != NULL)
	{
		clReleaseMemObject(this->floatingImageClmem);
		clReleaseMemObject(this->floMatClmem);
	}
	if(this->CurrentWarped != NULL)
		clReleaseMemObject(this->warpedImageClmem);
	if(this->CurrentDeformationField != NULL)
		clReleaseMemObject(this->deformationFieldClmem);
	if(this->CurrentReferenceMask != NULL)
		clReleaseMemObject(this->maskClmem);
	if(this->blockMatchingParams != NULL)
	{
		clReleaseMemObject(this->totalBlockClmem);
		clReleaseMemObject(this->referencePositionClmem);
		clReleaseMemObject(this->warpedPositionClmem);
	}
}
/* *************************************************************** */
bool ClAladinContent::isCurrentComputationDoubleCapable() {
	 return this->sContext->getIsCardDoubleCapable();
}
/* *************************************************************** */
