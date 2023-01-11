#include "CudaContent.h"

/* *************************************************************** */
CudaContent::CudaContent(nifti_image *referenceIn,
                         nifti_image *floatingIn,
                         int *referenceMaskIn,
                         mat44 *transformationMatrixIn,
                         size_t bytesIn) :
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateImages();
    AllocateWarped();
    AllocateDeformationField();
    SetReferenceMask(referenceMask);
    SetTransformationMatrix(transformationMatrix);
}
/* *************************************************************** */
CudaContent::~CudaContent() {
    DeallocateImages();
    DeallocateWarped();
    DeallocateDeformationField();
    SetReferenceMask(nullptr);
    SetTransformationMatrix(nullptr);
}
/* *************************************************************** */
void CudaContent::AllocateImages() {
    if (reference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(reference);
    if (floating->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(floating);
    if (reference->nt == 1) {
        cudaCommon_allocateArrayToDevice<float>(&referenceCuda[0], reference->dim);
        cudaCommon_transferNiftiToArrayOnDevice<float>(referenceCuda[0], reference);
        cudaCommon_allocateArrayToDevice<float>(&floatingCuda[0], floating->dim);
        cudaCommon_transferNiftiToArrayOnDevice<float>(floatingCuda[0], floating);
    } else if (reference->nt == 2) {
        cudaCommon_allocateArrayToDevice<float>(&referenceCuda[0], &referenceCuda[1], reference->dim);
        cudaCommon_transferNiftiToArrayOnDevice<float>(referenceCuda[0], referenceCuda[1], reference);
        cudaCommon_allocateArrayToDevice<float>(&floatingCuda[0], &floatingCuda[1], floating->dim);
        cudaCommon_transferNiftiToArrayOnDevice<float>(floatingCuda[0], floatingCuda[1], floating);
    }
}
/* *************************************************************** */
void CudaContent::DeallocateImages() {
    if (referenceCuda[0]) {
        cudaCommon_free(referenceCuda[0]);
        referenceCuda[0] = nullptr;
    }
    if (referenceCuda[1]) {
        cudaCommon_free(referenceCuda[1]);
        referenceCuda[1] = nullptr;
    }
    if (floatingCuda[0]) {
        cudaCommon_free(floatingCuda[0]);
        floatingCuda[0] = nullptr;
    }
    if (floatingCuda[1]) {
        cudaCommon_free(floatingCuda[1]);
        floatingCuda[1] = nullptr;
    }
}
/* *************************************************************** */
void CudaContent::AllocateDeformationField() {
    NR_CUDA_SAFE_CALL(cudaMalloc(&deformationFieldCuda, deformationField->nvox * sizeof(float4)));
}
/* *************************************************************** */
void CudaContent::DeallocateDeformationField() {
    if (deformationFieldCuda) {
        cudaCommon_free(deformationFieldCuda);
        deformationFieldCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaContent::AllocateWarped() {
    if (warped->nt == 1) {
        cudaCommon_allocateArrayToDevice<float>(&warpedCuda[0], warped->dim);
    } else if (warped->nt == 2) {
        cudaCommon_allocateArrayToDevice<float>(&warpedCuda[0], &warpedCuda[1], warped->dim);
    } else {
        reg_print_fct_error("CudaContent::AllocateWarped()");
        reg_print_msg_error("More than 2 time points aren't handled in the floating image");
        reg_exit();
    }
}
/* *************************************************************** */
void CudaContent::DeallocateWarped() {
    if (warpedCuda[0]) {
        cudaCommon_free(warpedCuda[0]);
        warpedCuda[0] = nullptr;
    }
    if (warpedCuda[1]) {
        cudaCommon_free(warpedCuda[1]);
        warpedCuda[1] = nullptr;
    }
}
/* *************************************************************** */
bool CudaContent::IsCurrentComputationDoubleCapable() {
    return CudaContextSingleton::Instance().GetIsCardDoubleCapable();
}
/* *************************************************************** */
nifti_image* CudaContent::GetDeformationField() {
    cudaCommon_transferFromDeviceToNifti(deformationField, deformationFieldCuda);
    return deformationField;
}
/* *************************************************************** */
void CudaContent::SetDeformationField(nifti_image *deformationFieldIn) {
    Content::SetDeformationField(deformationFieldIn);
    DeallocateDeformationField();
    if (!deformationField) return;

    AllocateDeformationField();
    cudaCommon_transferNiftiToArrayOnDevice(deformationFieldCuda, deformationField);
}
/* *************************************************************** */
void CudaContent::SetReferenceMask(int *referenceMaskIn) {
    Content::SetReferenceMask(referenceMaskIn);

    if (referenceMaskCuda) {
        cudaCommon_free(referenceMaskCuda);
        referenceMaskCuda = nullptr;
    }

    if (!referenceMask) return;

    NR_CUDA_SAFE_CALL(cudaMalloc(&referenceMaskCuda, reference->nvox * sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(referenceMaskCuda, referenceMask,
                                 reference->nvox * sizeof(int), cudaMemcpyHostToDevice));
}
/* *************************************************************** */
void CudaContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    Content::SetTransformationMatrix(transformationMatrixIn);

    if (transformationMatrixCuda) {
        cudaCommon_free(transformationMatrixCuda);
        transformationMatrixCuda = nullptr;
    }

    if (!transformationMatrix) return;

    float *transformationMatrixCptr = (float*)malloc(sizeof(mat44));
    mat44ToCptr(*transformationMatrix, transformationMatrixCptr);
    cudaCommon_allocateArrayToDevice(&transformationMatrixCuda, sizeof(mat44) / sizeof(float));
    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrixCuda, transformationMatrixCptr, sizeof(mat44), cudaMemcpyHostToDevice));
    free(transformationMatrixCptr);
}
/* *************************************************************** */
nifti_image* CudaContent::GetWarped(int index) {
    DownloadImage(warped, warpedCuda[index], warped->datatype);
    return warped;
}
/* *************************************************************** */
void CudaContent::SetWarped(nifti_image *warpedIn) {
    Content::SetWarped(warpedIn);
    DeallocateWarped();
    if (!warped) return;

    reg_tools_changeDatatype<float>(warped);
    AllocateWarped();
    cudaCommon_transferNiftiToArrayOnDevice(warpedCuda[0], warped);
    if (warpedCuda[1])
        cudaCommon_transferNiftiToArrayOnDevice(warpedCuda[1], warped);
}
/* *************************************************************** */
template<class DataType>
DataType CudaContent::CastImageData(float intensity, int datatype) {
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
template<class DataType>
void CudaContent::FillImageData(nifti_image *image, float *memoryObject, int datatype) {
    size_t size = image->nvox;
    float *buffer = (float*)malloc(size * sizeof(float));

    cudaCommon_transferFromDeviceToCpu(buffer, memoryObject, size);

    free(image->data);
    image->datatype = datatype;
    image->nbyper = sizeof(DataType);
    image->data = (void*)malloc(size * image->nbyper);
    DataType* data = static_cast<DataType*>(image->data);
    for (size_t i = 0; i < size; ++i)
        data[i] = CastImageData<DataType>(buffer[i], datatype);
    free(buffer);
}
/* *************************************************************** */
void CudaContent::DownloadImage(nifti_image *image, float *memoryObject, int datatype) {
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
        reg_print_fct_error("CudaContent::DownloadImage()");
        reg_print_msg_error("Unsupported type");
        break;
    }
}
/* *************************************************************** */
