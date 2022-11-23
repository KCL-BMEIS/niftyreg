#include "CLAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent() {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *currentReferenceIn,
                                 nifti_image *currentFloatingIn,
                                 int *currentReferenceMaskIn,
                                 size_t byte,
                                 const unsigned int blockPercentage,
                                 const unsigned int inlierLts,
                                 int blockStep) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  byte, blockPercentage,
                  inlierLts,
                  blockStep) {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *currentReferenceIn,
                                 nifti_image *currentFloatingIn,
                                 int *currentReferenceMaskIn,
                                 size_t byte) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  byte) {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *currentReferenceIn,
                                 nifti_image *currentFloatingIn,
                                 int *currentReferenceMaskIn,
                                 mat44 *transMat,
                                 size_t byte,
                                 const unsigned int blockPercentage,
                                 const unsigned int inlierLts,
                                 int blockStep) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  transMat,
                  byte,
                  blockPercentage,
                  inlierLts,
                  blockStep) {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *currentReferenceIn,
                                 nifti_image *currentFloatingIn,
                                 int *currentReferenceMaskIn,
                                 mat44 *transMat,
                                 size_t byte) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  transMat,
                  byte) {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::~ClAladinContent() {
    FreeClPtrs();
}
/* *************************************************************** */
void ClAladinContent::InitVars() {
    this->referenceImageClmem = 0;
    this->floatingImageClmem = 0;
    this->warpedImageClmem = 0;
    this->deformationFieldClmem = 0;
    this->referencePositionClmem = 0;
    this->warpedPositionClmem = 0;
    this->totalBlockClmem = 0;
    this->maskClmem = 0;

    if (this->currentReference != nullptr && this->currentReference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(this->currentReference);
    if (this->currentFloating != nullptr && this->currentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(this->currentFloating);
        if (this->currentWarped != nullptr)
            reg_tools_changeDatatype<float>(this->currentWarped);
    }
    this->sContext = &ClContextSingleton::Instance();
    this->clContext = this->sContext->GetContext();
    this->commandQueue = this->sContext->GetCommandQueue();
    //this->numBlocks = (this->blockMatchingParams != nullptr) ? this->blockMatchingParams->blockNumber[0] * this->blockMatchingParams->blockNumber[1] * this->blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void ClAladinContent::AllocateClPtrs() {

    if (this->currentWarped != nullptr) {
        this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->currentWarped->nvox * sizeof(float), this->currentWarped->data, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (warpedImageClmem): ");
    }
    if (this->currentDeformationField != nullptr) {
        this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->currentDeformationField->nvox, this->currentDeformationField->data, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (deformationFieldClmem): ");
    }
    if (this->currentFloating != nullptr) {
        this->floatingImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * this->currentFloating->nvox, this->currentFloating->data, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (currentFloating): ");

        float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
        mat44ToCptr(this->floMatrix_ijk, sourceIJKMatrix_h);
        this->floMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           16 * sizeof(float),
                                           sourceIJKMatrix_h, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (floMatClmem): ");
        free(sourceIJKMatrix_h);
    }
    if (this->currentReference != nullptr) {
        this->referenceImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   sizeof(float) * this->currentReference->nvox,
                                                   this->currentReference->data, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (referenceImageClmem): ");

        float* targetMat = (float *)malloc(16 * sizeof(float)); //freed
        mat44ToCptr(this->refMatrix_xyz, targetMat);
        this->refMatClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           16 * sizeof(float),
                                           targetMat, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (refMatClmem): ");
        free(targetMat);
    }
    if (this->blockMatchingParams != nullptr) {
        if (this->blockMatchingParams->referencePosition != nullptr) {
            //targetPositionClmem
            this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                          this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
                                                          this->blockMatchingParams->referencePosition, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (referencePositionClmem): ");
        }
        if (this->blockMatchingParams->warpedPosition != nullptr) {
            //resultPositionClmem
            this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                       this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float),
                                                       this->blockMatchingParams->warpedPosition, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (warpedPositionClmem): ");
        }
        if (this->blockMatchingParams->totalBlock != nullptr) {
            //totalBlockClmem
            this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   this->blockMatchingParams->totalBlockNumber * sizeof(int),
                                                   this->blockMatchingParams->totalBlock, &this->errNum);
            this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (activeBlockClmem): ");
        }
    }
    if (this->currentReferenceMask != nullptr && this->currentReference != nullptr) {
        this->maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         this->currentReference->nx * this->currentReference->ny * this->currentReference->nz * sizeof(int),
                                         this->currentReferenceMask, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClContent::AllocateClPtrs failed to allocate memory (clCreateBuffer): ");
    }
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetCurrentWarped(int datatype) {
    DownloadImage(this->currentWarped, this->warpedImageClmem, datatype);
    return this->currentWarped;
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetCurrentDeformationField() {
    this->errNum = clEnqueueReadBuffer(this->commandQueue, this->deformationFieldClmem, CL_TRUE, 0, this->currentDeformationField->nvox * sizeof(float), this->currentDeformationField->data, 0, nullptr, nullptr); //CLCONTEXT
    this->sContext->checkErrNum(errNum, "Get: failed currentDeformationField: ");
    return this->currentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* ClAladinContent::GetBlockMatchingParams() {
    this->errNum = clEnqueueReadBuffer(this->commandQueue, this->warpedPositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->warpedPosition, 0, nullptr, nullptr); //CLCONTEXT
    this->sContext->checkErrNum(this->errNum, "CLContext: failed result position: ");
    this->errNum = clEnqueueReadBuffer(this->commandQueue, this->referencePositionClmem, CL_TRUE, 0, sizeof(float) * this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim, this->blockMatchingParams->referencePosition, 0, nullptr, nullptr); //CLCONTEXT
    this->sContext->checkErrNum(this->errNum, "CLContext: failed target position: ");
    return this->blockMatchingParams;
}
/* *************************************************************** */
void ClAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    AladinContent::SetTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::SetCurrentDeformationField(nifti_image *currentDeformationFieldIn) {
    if (this->currentDeformationField != nullptr)
        clReleaseMemObject(this->deformationFieldClmem);

    AladinContent::SetCurrentDeformationField(currentDeformationFieldIn);
    this->deformationFieldClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->currentDeformationField->nvox * sizeof(float), this->currentDeformationField->data, &this->errNum);
    this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetCurrentDeformationField failed to allocate memory (deformationFieldClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetCurrentReferenceMask(int *maskIn, size_t nvox) {
    if (this->currentReferenceMask != nullptr)
        clReleaseMemObject(maskClmem);
    this->currentReferenceMask = maskIn;
    this->maskClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvox * sizeof(int), this->currentReferenceMask, &this->errNum);
    this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetCurrentReferenceMask failed to allocate memory (maskClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetCurrentWarped(nifti_image *currentWarped) {
    if (this->currentWarped != nullptr) {
        clReleaseMemObject(this->warpedImageClmem);
    }
    if (currentWarped->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(currentWarped);
    }
    AladinContent::SetCurrentWarped(currentWarped);
    this->warpedImageClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->currentWarped->nvox * sizeof(float), this->currentWarped->data, &this->errNum);
    this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetCurrentWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {

    AladinContent::SetBlockMatchingParams(bmp);
    if (this->blockMatchingParams->referencePosition != nullptr) {
        clReleaseMemObject(this->referencePositionClmem);
        //referencePositionClmem
        this->referencePositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->referencePosition, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
    }
    if (this->blockMatchingParams->warpedPosition != nullptr) {
        clReleaseMemObject(this->warpedPositionClmem);
        //warpedPositionClmem
        this->warpedPositionClmem = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->activeBlockNumber * this->blockMatchingParams->dim * sizeof(float), this->blockMatchingParams->warpedPosition, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
    }
    if (this->blockMatchingParams->totalBlock != nullptr) {
        clReleaseMemObject(this->totalBlockClmem);
        //totalBlockClmem
        this->totalBlockClmem = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, this->blockMatchingParams->totalBlockNumber * sizeof(int), this->blockMatchingParams->totalBlock, &this->errNum);
        this->sContext->checkErrNum(this->errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (activeBlockClmem): ");
    }
}
/* *************************************************************** */
cl_mem ClAladinContent::GetReferenceImageArrayClmem() {
    return this->referenceImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetFloatingImageArrayClmem() {
    return this->floatingImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetWarpedImageClmem() {
    return this->warpedImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetReferencePositionClmem() {
    return this->referencePositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetWarpedPositionClmem() {
    return this->warpedPositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetDeformationFieldArrayClmem() {
    return this->deformationFieldClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetTotalBlockClmem() {
    return this->totalBlockClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetMaskClmem() {
    return this->maskClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetRefMatClmem() {
    return this->refMatClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetFloMatClmem() {
    return this->floMatClmem;
}
/* *************************************************************** */
int *ClAladinContent::GetReferenceDims() {
    return this->referenceDims;
}
/* *************************************************************** */
int *ClAladinContent::GetFloatingDims() {
    return this->floatingDims;
}
/* *************************************************************** */
template<class DataType>
DataType ClAladinContent::FillWarpedImageData(float intensity, int datatype) {
    switch (datatype) {
    case NIFTI_TYPE_FLOAT32:
        return static_cast<float>(intensity);
        break;
    case NIFTI_TYPE_FLOAT64:
        return static_cast<double>(intensity);
        break;
    case NIFTI_TYPE_UINT8:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 255 ? reg_round(intensity) : 255); // 255=2^8-1
        return static_cast<unsigned char>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT16:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 65535 ? reg_round(intensity) : 65535); // 65535=2^16-1
        return static_cast<unsigned short>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT32:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 4294967295 ? reg_round(intensity) : 4294967295); // 4294967295=2^32-1
        return static_cast<unsigned int>(intensity > 0 ? reg_round(intensity) : 0);
        break;
    default:
        if (intensity != intensity)
            intensity = 0;
        return static_cast<DataType>(reg_round(intensity));
        break;
    }
}
/* *************************************************************** */
template<class T>
void ClAladinContent::FillImageData(nifti_image *image,
                                    cl_mem memoryObject,
                                    int type) {
    size_t size = image->nvox;
    float* buffer = nullptr;
    buffer = (float*)malloc(size * sizeof(float));
    if (buffer == nullptr) {
        reg_print_fct_error("ClAladinContent::FillImageData");
        reg_print_msg_error("Memory allocation did not complete successfully. Exit.");
        reg_exit();
    }

    this->errNum = clEnqueueReadBuffer(this->commandQueue, memoryObject, CL_TRUE, 0,
                                       size * sizeof(float), buffer, 0, nullptr, nullptr);
    this->sContext->checkErrNum(this->errNum, "Error reading warped buffer.");

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = (void *)malloc(image->nvox * image->nbyper);
    T* dataT = static_cast<T*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = FillWarpedImageData<T>(buffer[i], type);
    free(buffer);
}
/* *************************************************************** */
void ClAladinContent::DownloadImage(nifti_image *image,
                                    cl_mem memoryObject,
                                    int datatype) {
    switch (datatype) {
    case NIFTI_TYPE_FLOAT32:
        FillImageData<float>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_FLOAT64:
        FillImageData<double>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT8:
        FillImageData<unsigned char>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT8:
        FillImageData<char>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT16:
        FillImageData<unsigned short>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT16:
        FillImageData<short>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_UINT32:
        FillImageData<unsigned int>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT32:
        FillImageData<int>(image, memoryObject, datatype);
        break;
    default:
        reg_print_fct_error("ClAladinContent::DownloadImage");
        reg_print_msg_error("Unsupported type");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
void ClAladinContent::FreeClPtrs() {
    if (this->currentReference != nullptr) {
        clReleaseMemObject(this->referenceImageClmem);
        clReleaseMemObject(this->refMatClmem);
    }
    if (this->currentFloating != nullptr) {
        clReleaseMemObject(this->floatingImageClmem);
        clReleaseMemObject(this->floMatClmem);
    }
    if (this->currentWarped != nullptr)
        clReleaseMemObject(this->warpedImageClmem);
    if (this->currentDeformationField != nullptr)
        clReleaseMemObject(this->deformationFieldClmem);
    if (this->currentReferenceMask != nullptr)
        clReleaseMemObject(this->maskClmem);
    if (this->blockMatchingParams != nullptr) {
        clReleaseMemObject(this->totalBlockClmem);
        clReleaseMemObject(this->referencePositionClmem);
        clReleaseMemObject(this->warpedPositionClmem);
    }
}
/* *************************************************************** */
bool ClAladinContent::IsCurrentComputationDoubleCapable() {
    return this->sContext->GetIsCardDoubleCapable();
}
/* *************************************************************** */
