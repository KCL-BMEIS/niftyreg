#include "CLAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *currentReferenceIn,
                                 nifti_image *currentFloatingIn,
                                 int *currentReferenceMaskIn,
                                 mat44 *transformationMatrixIn,
                                 size_t bytesIn,
                                 const unsigned int percentageOfBlocks,
                                 const unsigned int inlierLts,
                                 int blockStepSize) :
    AladinContent(currentReferenceIn,
                  currentFloatingIn,
                  currentReferenceMaskIn,
                  transformationMatrixIn,
                  bytesIn,
                  percentageOfBlocks,
                  inlierLts,
                  blockStepSize) {
    InitVars();
    AllocateClPtrs();
}
/* *************************************************************** */
ClAladinContent::~ClAladinContent() {
    FreeClPtrs();
}
/* *************************************************************** */
void ClAladinContent::InitVars() {
    referenceImageClmem = nullptr;
    floatingImageClmem = nullptr;
    warpedImageClmem = nullptr;
    deformationFieldClmem = nullptr;
    referencePositionClmem = nullptr;
    warpedPositionClmem = nullptr;
    totalBlockClmem = nullptr;
    maskClmem = nullptr;

    if (currentReference != nullptr && currentReference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(currentReference);
    if (currentFloating != nullptr && currentFloating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(currentFloating);
        if (currentWarped != nullptr)
            reg_tools_changeDatatype<float>(currentWarped);
    }
    sContext = &ClContextSingleton::Instance();
    clContext = sContext->GetContext();
    commandQueue = sContext->GetCommandQueue();
    //numBlocks = (blockMatchingParams != nullptr) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void ClAladinContent::AllocateClPtrs() {
    if (currentWarped != nullptr) {
        warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, currentWarped->nvox * sizeof(float), currentWarped->data, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (warpedImageClmem): ");
    }
    if (currentDeformationField != nullptr) {
        deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * currentDeformationField->nvox, currentDeformationField->data, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (deformationFieldClmem): ");
    }
    if (currentFloating != nullptr) {
        floatingImageClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * currentFloating->nvox, currentFloating->data, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (currentFloating): ");

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(currentFloating), sourceIJKMatrix_h);
        floMatClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mat44), sourceIJKMatrix_h, &errNum);
        sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (floMatClmem): ");
        free(sourceIJKMatrix_h);
    }
    if (currentReference != nullptr) {
        referenceImageClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             sizeof(float) * currentReference->nvox,
                                             currentReference->data, &errNum);
        sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (referenceImageClmem): ");

        float* targetMat = (float *)malloc(sizeof(mat44)); //freed
        mat44ToCptr(*GetXYZMatrix(currentReference), targetMat);
        refMatClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mat44), targetMat, &errNum);
        sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (refMatClmem): ");
        free(targetMat);
    }
    if (blockMatchingParams != nullptr) {
        if (blockMatchingParams->referencePosition != nullptr) {
            //targetPositionClmem
            referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                    blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                    blockMatchingParams->referencePosition, &errNum);
            sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (referencePositionClmem): ");
        }
        if (blockMatchingParams->warpedPosition != nullptr) {
            //resultPositionClmem
            warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                 blockMatchingParams->warpedPosition, &errNum);
            sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (warpedPositionClmem): ");
        }
        if (blockMatchingParams->totalBlock != nullptr) {
            //totalBlockClmem
            totalBlockClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             blockMatchingParams->totalBlockNumber * sizeof(int),
                                             blockMatchingParams->totalBlock, &errNum);
            sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (activeBlockClmem): ");
        }
    }
    if (currentReferenceMask != nullptr && currentReference != nullptr) {
        maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   currentReference->nx * currentReference->ny * currentReference->nz * sizeof(int),
                                   currentReferenceMask, &errNum);
        sContext->checkErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (clCreateBuffer): ");
    }
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetCurrentWarped(int datatype) {
    DownloadImage(currentWarped, warpedImageClmem, datatype);
    return currentWarped;
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetCurrentDeformationField() {
    errNum = clEnqueueReadBuffer(commandQueue, deformationFieldClmem, CL_TRUE, 0, currentDeformationField->nvox * sizeof(float), currentDeformationField->data, 0, nullptr, nullptr); //CLCONTEXT
    sContext->checkErrNum(errNum, "Get: failed currentDeformationField: ");
    return currentDeformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* ClAladinContent::GetBlockMatchingParams() {
    errNum = clEnqueueReadBuffer(commandQueue, warpedPositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * blockMatchingParams->dim, blockMatchingParams->warpedPosition, 0, nullptr, nullptr); //CLCONTEXT
    sContext->checkErrNum(errNum, "CLContext: failed result position: ");
    errNum = clEnqueueReadBuffer(commandQueue, referencePositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * blockMatchingParams->dim, blockMatchingParams->referencePosition, 0, nullptr, nullptr); //CLCONTEXT
    sContext->checkErrNum(errNum, "CLContext: failed target position: ");
    return blockMatchingParams;
}
/* *************************************************************** */
void ClAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    AladinContent::SetTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::SetCurrentDeformationField(nifti_image *currentDeformationFieldIn) {
    if (currentDeformationField != nullptr)
        clReleaseMemObject(deformationFieldClmem);

    AladinContent::SetCurrentDeformationField(currentDeformationFieldIn);
    deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, currentDeformationField->nvox * sizeof(float), currentDeformationField->data, &errNum);
    sContext->checkErrNum(errNum, "ClAladinContent::SetCurrentDeformationField failed to allocate memory (deformationFieldClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetCurrentReferenceMask(int *currentReferenceMaskIn) {
    if (currentReferenceMask != nullptr)
        clReleaseMemObject(maskClmem);
    AladinContent::SetCurrentReferenceMask(currentReferenceMaskIn);
    maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, currentReference->nvox * sizeof(int), currentReferenceMask, &errNum);
    sContext->checkErrNum(errNum, "ClAladinContent::SetCurrentReferenceMask failed to allocate memory (maskClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetCurrentWarped(nifti_image *currentWarped) {
    if (currentWarped != nullptr) {
        clReleaseMemObject(warpedImageClmem);
    }
    if (currentWarped->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(currentWarped);
    }
    AladinContent::SetCurrentWarped(currentWarped);
    warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, currentWarped->nvox * sizeof(float), currentWarped->data, &errNum);
    sContext->checkErrNum(errNum, "ClAladinContent::SetCurrentWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {

    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition != nullptr) {
        clReleaseMemObject(referencePositionClmem);
        //referencePositionClmem
        referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->referencePosition, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
    }
    if (blockMatchingParams->warpedPosition != nullptr) {
        clReleaseMemObject(warpedPositionClmem);
        //warpedPositionClmem
        warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->warpedPosition, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
    }
    if (blockMatchingParams->totalBlock != nullptr) {
        clReleaseMemObject(totalBlockClmem);
        //totalBlockClmem
        totalBlockClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, blockMatchingParams->totalBlockNumber * sizeof(int), blockMatchingParams->totalBlock, &errNum);
        sContext->checkErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (activeBlockClmem): ");
    }
}
/* *************************************************************** */
cl_mem ClAladinContent::GetReferenceImageArrayClmem() {
    return referenceImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetFloatingImageArrayClmem() {
    return floatingImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetWarpedImageClmem() {
    return warpedImageClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetReferencePositionClmem() {
    return referencePositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetWarpedPositionClmem() {
    return warpedPositionClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetDeformationFieldArrayClmem() {
    return deformationFieldClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetTotalBlockClmem() {
    return totalBlockClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetMaskClmem() {
    return maskClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetRefMatClmem() {
    return refMatClmem;
}
/* *************************************************************** */
cl_mem ClAladinContent::GetFloMatClmem() {
    return floMatClmem;
}
/* *************************************************************** */
int *ClAladinContent::GetReferenceDims() {
    return referenceDims;
}
/* *************************************************************** */
int *ClAladinContent::GetFloatingDims() {
    return floatingDims;
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

    errNum = clEnqueueReadBuffer(commandQueue, memoryObject, CL_TRUE, 0,
                                 size * sizeof(float), buffer, 0, nullptr, nullptr);
    sContext->checkErrNum(errNum, "Error reading warped buffer.");

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
    if (currentReference != nullptr) {
        clReleaseMemObject(referenceImageClmem);
        clReleaseMemObject(refMatClmem);
    }
    if (currentFloating != nullptr) {
        clReleaseMemObject(floatingImageClmem);
        clReleaseMemObject(floMatClmem);
    }
    if (currentWarped != nullptr)
        clReleaseMemObject(warpedImageClmem);
    if (currentDeformationField != nullptr)
        clReleaseMemObject(deformationFieldClmem);
    if (currentReferenceMask != nullptr)
        clReleaseMemObject(maskClmem);
    if (blockMatchingParams != nullptr) {
        clReleaseMemObject(totalBlockClmem);
        clReleaseMemObject(referencePositionClmem);
        clReleaseMemObject(warpedPositionClmem);
    }
}
/* *************************************************************** */
bool ClAladinContent::IsCurrentComputationDoubleCapable() {
    return sContext->GetIsCardDoubleCapable();
}
/* *************************************************************** */
