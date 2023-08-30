#include "ClAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent(nifti_image *referenceIn,
                                 nifti_image *floatingIn,
                                 int *referenceMaskIn,
                                 mat44 *transformationMatrixIn,
                                 size_t bytesIn,
                                 const unsigned percentageOfBlocks,
                                 const unsigned inlierLts,
                                 int blockStepSize) :
    AladinContent(referenceIn,
                  floatingIn,
                  referenceMaskIn,
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

    if (reference != nullptr && reference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(reference);
    if (floating != nullptr && floating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(floating);
        if (warped != nullptr)
            reg_tools_changeDatatype<float>(warped);
    }
    sContext = &ClContextSingleton::GetInstance();
    clContext = sContext->GetContext();
    commandQueue = sContext->GetCommandQueue();
    //numBlocks = (blockMatchingParams != nullptr) ? blockMatchingParams->blockNumber[0] * blockMatchingParams->blockNumber[1] * blockMatchingParams->blockNumber[2] : 0;
}
/* *************************************************************** */
void ClAladinContent::AllocateClPtrs() {
    if (warped != nullptr) {
        warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, warped->nvox * sizeof(float), warped->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (warpedImageClmem): ");
    }
    if (deformationField != nullptr) {
        deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * deformationField->nvox, deformationField->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (deformationFieldClmem): ");
    }
    if (floating != nullptr) {
        floatingImageClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * floating->nvox, floating->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (floating): ");

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(*floating), sourceIJKMatrix_h);
        floMatClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mat44), sourceIJKMatrix_h, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (floMatClmem): ");
        free(sourceIJKMatrix_h);
    }
    if (reference != nullptr) {
        referenceImageClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             sizeof(float) * reference->nvox,
                                             reference->data, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (referenceImageClmem): ");

        float* targetMat = (float *)malloc(sizeof(mat44)); //freed
        mat44ToCptr(*GetXYZMatrix(*reference), targetMat);
        refMatClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mat44), targetMat, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (refMatClmem): ");
        free(targetMat);
    }
    if (blockMatchingParams != nullptr) {
        if (blockMatchingParams->referencePosition != nullptr) {
            //targetPositionClmem
            referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                    blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                    blockMatchingParams->referencePosition, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (referencePositionClmem): ");
        }
        if (blockMatchingParams->warpedPosition != nullptr) {
            //resultPositionClmem
            warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                 blockMatchingParams->warpedPosition, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (warpedPositionClmem): ");
        }
        if (blockMatchingParams->totalBlock != nullptr) {
            //totalBlockClmem
            totalBlockClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             blockMatchingParams->totalBlockNumber * sizeof(int),
                                             blockMatchingParams->totalBlock, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (activeBlockClmem): ");
        }
    }
    if (referenceMask != nullptr && reference != nullptr) {
        maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   NiftiImage::calcVoxelNumber(reference, 3) * sizeof(int), referenceMask, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (clCreateBuffer): ");
    }
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetWarped() {
    DownloadImage(warped, warpedImageClmem, warped->datatype);
    return warped;
}
/* *************************************************************** */
nifti_image* ClAladinContent::GetDeformationField() {
    errNum = clEnqueueReadBuffer(commandQueue, deformationFieldClmem, CL_TRUE, 0, deformationField->nvox * sizeof(float), deformationField->data, 0, nullptr, nullptr); //CLCONTEXT
    sContext->CheckErrNum(errNum, "Get: failed deformationField: ");
    return deformationField;
}
/* *************************************************************** */
_reg_blockMatchingParam* ClAladinContent::GetBlockMatchingParams() {
    errNum = clEnqueueReadBuffer(commandQueue, warpedPositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * blockMatchingParams->dim, blockMatchingParams->warpedPosition, 0, nullptr, nullptr); //CLCONTEXT
    sContext->CheckErrNum(errNum, "CLContext: failed result position: ");
    errNum = clEnqueueReadBuffer(commandQueue, referencePositionClmem, CL_TRUE, 0, sizeof(float) * blockMatchingParams->activeBlockNumber * blockMatchingParams->dim, blockMatchingParams->referencePosition, 0, nullptr, nullptr); //CLCONTEXT
    sContext->CheckErrNum(errNum, "CLContext: failed target position: ");
    return blockMatchingParams;
}
/* *************************************************************** */
void ClAladinContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    AladinContent::SetTransformationMatrix(transformationMatrixIn);
}
/* *************************************************************** */
void ClAladinContent::SetDeformationField(nifti_image *deformationFieldIn) {
    if (deformationField != nullptr)
        clReleaseMemObject(deformationFieldClmem);

    AladinContent::SetDeformationField(deformationFieldIn);
    deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, deformationField->nvox * sizeof(float), deformationField->data, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetDeformationField failed to allocate memory (deformationFieldClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetReferenceMask(int *referenceMaskIn) {
    if (referenceMask != nullptr)
        clReleaseMemObject(maskClmem);
    AladinContent::SetReferenceMask(referenceMaskIn);
    maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, reference->nvox * sizeof(int), referenceMask, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetReferenceMask failed to allocate memory (maskClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetWarped(nifti_image *warped) {
    if (warped != nullptr) {
        clReleaseMemObject(warpedImageClmem);
    }
    if (warped->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(warped);
    }
    AladinContent::SetWarped(warped);
    warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, warped->nvox * sizeof(float), warped->data, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition != nullptr) {
        clReleaseMemObject(referencePositionClmem);
        //referencePositionClmem
        referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->referencePosition, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
    }
    if (blockMatchingParams->warpedPosition != nullptr) {
        clReleaseMemObject(warpedPositionClmem);
        //warpedPositionClmem
        warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->warpedPosition, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
    }
    if (blockMatchingParams->totalBlock != nullptr) {
        clReleaseMemObject(totalBlockClmem);
        //totalBlockClmem
        totalBlockClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, blockMatchingParams->totalBlockNumber * sizeof(int), blockMatchingParams->totalBlock, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (activeBlockClmem): ");
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
        return static_cast<DataType>(intensity);
    case NIFTI_TYPE_FLOAT64:
        return static_cast<DataType>(intensity);
    case NIFTI_TYPE_UINT8:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
        return static_cast<unsigned char>(intensity > 0 ? Round(intensity) : 0);
    case NIFTI_TYPE_UINT16:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
        return static_cast<unsigned short>(intensity > 0 ? Round(intensity) : 0);
    case NIFTI_TYPE_UINT32:
        if (intensity != intensity)
            intensity = 0;
        intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
        return static_cast<unsigned>(intensity > 0 ? Round(intensity) : 0);
    default:
        if (intensity != intensity)
            intensity = 0;
        return static_cast<DataType>(Round(intensity));
    }
}
/* *************************************************************** */
template<class T>
void ClAladinContent::FillImageData(nifti_image *image, cl_mem memoryObject, int type) {
    size_t size = image->nvox;
    float* buffer = nullptr;
    buffer = (float*)malloc(size * sizeof(float));
    if (buffer == nullptr)
        NR_FATAL_ERROR("Memory allocation did not complete successfully");

    errNum = clEnqueueReadBuffer(commandQueue, memoryObject, CL_TRUE, 0,
                                 size * sizeof(float), buffer, 0, nullptr, nullptr);
    sContext->CheckErrNum(errNum, "Error reading warped buffer.");

    free(image->data);
    image->datatype = type;
    image->nbyper = sizeof(T);
    image->data = malloc(image->nvox * image->nbyper);
    T* dataT = static_cast<T*>(image->data);
    for (size_t i = 0; i < size; ++i)
        dataT[i] = FillWarpedImageData<T>(buffer[i], type);
    free(buffer);
}
/* *************************************************************** */
void ClAladinContent::DownloadImage(nifti_image *image, cl_mem memoryObject, int datatype) {
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
        FillImageData<unsigned>(image, memoryObject, datatype);
        break;
    case NIFTI_TYPE_INT32:
        FillImageData<int>(image, memoryObject, datatype);
        break;
    default:
        NR_FATAL_ERROR("Unsupported type");
    }
}
/* *************************************************************** */
void ClAladinContent::FreeClPtrs() {
    if (reference != nullptr) {
        clReleaseMemObject(referenceImageClmem);
        clReleaseMemObject(refMatClmem);
    }
    if (floating != nullptr) {
        clReleaseMemObject(floatingImageClmem);
        clReleaseMemObject(floMatClmem);
    }
    if (warped != nullptr)
        clReleaseMemObject(warpedImageClmem);
    if (deformationField != nullptr)
        clReleaseMemObject(deformationFieldClmem);
    if (referenceMask != nullptr)
        clReleaseMemObject(maskClmem);
    if (blockMatchingParams != nullptr) {
        clReleaseMemObject(totalBlockClmem);
        clReleaseMemObject(referencePositionClmem);
        clReleaseMemObject(warpedPositionClmem);
    }
}
/* *************************************************************** */
bool ClAladinContent::IsCurrentComputationDoubleCapable() {
    return sContext->IsCardDoubleCapable();
}
/* *************************************************************** */
