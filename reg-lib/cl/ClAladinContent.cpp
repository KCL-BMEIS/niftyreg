#include "ClAladinContent.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAladinContent::ClAladinContent(NiftiImage& referenceIn,
                                 NiftiImage& floatingIn,
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

    if (reference && reference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(reference);
    if (floating && floating->nbyper != NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<float>(floating);
        if (warped)
            reg_tools_changeDatatype<float>(warped);
    }
    sContext = &ClContextSingleton::GetInstance();
    clContext = sContext->GetContext();
    commandQueue = sContext->GetCommandQueue();
}
/* *************************************************************** */
void ClAladinContent::AllocateClPtrs() {
    if (warped) {
        warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, warped->nvox * sizeof(float), warped->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (warpedImageClmem): ");
    }
    if (deformationField) {
        deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * deformationField->nvox, deformationField->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (deformationFieldClmem): ");
    }
    if (floating) {
        floatingImageClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * floating->nvox, floating->data, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::AllocateClPtrs failed to allocate memory (floating): ");

        float *sourceIJKMatrix_h = (float*)malloc(sizeof(mat44));
        mat44ToCptr(*GetIJKMatrix(*floating), sourceIJKMatrix_h);
        floMatClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mat44), sourceIJKMatrix_h, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (floMatClmem): ");
        free(sourceIJKMatrix_h);
    }
    if (reference) {
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
    if (blockMatchingParams) {
        if (blockMatchingParams->referencePosition) {
            //targetPositionClmem
            referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                    blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                    blockMatchingParams->referencePosition, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (referencePositionClmem): ");
        }
        if (blockMatchingParams->warpedPosition) {
            //resultPositionClmem
            warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float),
                                                 blockMatchingParams->warpedPosition, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (warpedPositionClmem): ");
        }
        if (blockMatchingParams->totalBlock) {
            //totalBlockClmem
            totalBlockClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             blockMatchingParams->totalBlockNumber * sizeof(int),
                                             blockMatchingParams->totalBlock, &errNum);
            sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (activeBlockClmem): ");
        }
    }
    if (referenceMask && reference) {
        maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   reference.nVoxelsPerVolume() * sizeof(int), referenceMask, &errNum);
        sContext->CheckErrNum(errNum, "ClContent::AllocateClPtrs failed to allocate memory (clCreateBuffer): ");
    }
}
/* *************************************************************** */
NiftiImage& ClAladinContent::GetWarped() {
    DownloadImage(warped, warpedImageClmem, warped->datatype);
    return warped;
}
/* *************************************************************** */
NiftiImage& ClAladinContent::GetDeformationField() {
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
void ClAladinContent::SetDeformationField(NiftiImage&& deformationFieldIn) {
    if (deformationField)
        clReleaseMemObject(deformationFieldClmem);

    AladinContent::SetDeformationField(std::move(deformationFieldIn));
    deformationFieldClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, deformationField->nvox * sizeof(float), deformationField->data, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetDeformationField failed to allocate memory (deformationFieldClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetReferenceMask(int *referenceMaskIn) {
    if (referenceMask)
        clReleaseMemObject(maskClmem);
    AladinContent::SetReferenceMask(referenceMaskIn);
    maskClmem = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, reference->nvox * sizeof(int), referenceMask, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetReferenceMask failed to allocate memory (maskClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetWarped(NiftiImage&& warpedIn) {
    if (warpedIn->nbyper != NIFTI_TYPE_FLOAT32)
        warpedIn.changeDatatype(NIFTI_TYPE_FLOAT32);
    if (warped)
        clReleaseMemObject(warpedImageClmem);
    AladinContent::SetWarped(std::move(warpedIn));
    warpedImageClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, warped->nvox * sizeof(float), warped->data, &errNum);
    sContext->CheckErrNum(errNum, "ClAladinContent::SetWarped failed to allocate memory (warpedImageClmem): ");
}
/* *************************************************************** */
void ClAladinContent::SetBlockMatchingParams(_reg_blockMatchingParam* bmp) {
    AladinContent::SetBlockMatchingParams(bmp);
    if (blockMatchingParams->referencePosition) {
        clReleaseMemObject(referencePositionClmem);
        //referencePositionClmem
        referencePositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->referencePosition, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (referencePositionClmem): ");
    }
    if (blockMatchingParams->warpedPosition) {
        clReleaseMemObject(warpedPositionClmem);
        //warpedPositionClmem
        warpedPositionClmem = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockMatchingParams->activeBlockNumber * blockMatchingParams->dim * sizeof(float), blockMatchingParams->warpedPosition, &errNum);
        sContext->CheckErrNum(errNum, "ClAladinContent::SetBlockMatchingParams failed to allocate memory (warpedPositionClmem): ");
    }
    if (blockMatchingParams->totalBlock) {
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
void ClAladinContent::DownloadImage(NiftiImage& image, cl_mem memoryObject, int datatype) {
    const size_t size = image->nvox;
    unique_ptr<float[]> buffer(new float[size]);

    errNum = clEnqueueReadBuffer(commandQueue, memoryObject, CL_TRUE, 0,
                                 size * sizeof(float), buffer.get(), 0, nullptr, nullptr);
    sContext->CheckErrNum(errNum, "Error reading warped buffer.");

    std::visit([&](auto&& dataType) {
        using DataType = std::decay_t<decltype(dataType)>;
        image->datatype = datatype;
        image->nbyper = sizeof(DataType);
        image.realloc();
        DataType *data = static_cast<DataType*>(image->data);
        for (size_t i = 0; i < size; ++i)
            data[i] = image.clampData<DataType>(buffer[i]);
    }, image.getDataType());
}
/* *************************************************************** */
void ClAladinContent::FreeClPtrs() {
    if (reference) {
        clReleaseMemObject(referenceImageClmem);
        clReleaseMemObject(refMatClmem);
    }
    if (floating) {
        clReleaseMemObject(floatingImageClmem);
        clReleaseMemObject(floMatClmem);
    }
    if (warped)
        clReleaseMemObject(warpedImageClmem);
    if (deformationField)
        clReleaseMemObject(deformationFieldClmem);
    if (referenceMask)
        clReleaseMemObject(maskClmem);
    if (blockMatchingParams) {
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
