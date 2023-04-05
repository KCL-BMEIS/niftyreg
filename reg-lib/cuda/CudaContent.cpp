#include "CudaContent.h"

/* *************************************************************** */
CudaContent::CudaContent(nifti_image *referenceIn,
                         nifti_image *floatingIn,
                         int *referenceMaskIn,
                         mat44 *transformationMatrixIn,
                         size_t bytesIn):
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
    cudaCommon_allocateArrayToDevice<float>(&referenceCuda, reference->dim);
    cudaCommon_transferNiftiToArrayOnDevice<float>(referenceCuda, reference);
    cudaCommon_allocateArrayToDevice<float>(&floatingCuda, floating->dim);
    cudaCommon_transferNiftiToArrayOnDevice<float>(floatingCuda, floating);
}
/* *************************************************************** */
void CudaContent::DeallocateImages() {
    if (referenceCuda) {
        cudaCommon_free(referenceCuda);
        referenceCuda = nullptr;
    }
    if (floatingCuda) {
        cudaCommon_free(floatingCuda);
        floatingCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaContent::AllocateDeformationField() {
    cudaCommon_allocateArrayToDevice(&deformationFieldCuda, deformationField->dim);
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
    cudaCommon_allocateArrayToDevice<float>(&warpedCuda, warped->dim);
}
/* *************************************************************** */
void CudaContent::DeallocateWarped() {
    if (warpedCuda) {
        cudaCommon_free(warpedCuda);
        warpedCuda = nullptr;
    }
}
/* *************************************************************** */
bool CudaContent::IsCurrentComputationDoubleCapable() {
    return NiftyReg::CudaContext::GetInstance().IsCardDoubleCapable();
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
void CudaContent::UpdateDeformationField() {
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

    int *targetMask;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&targetMask, reference->nvox * sizeof(*targetMask)));
    int *targetMaskPtr = targetMask;
    activeVoxelNumber = 0;
    for (size_t i = 0; i < reference->nvox; i++) {
        if (referenceMask[i] != -1) {
            *targetMaskPtr++ = i;
            activeVoxelNumber++;
        }
    }

    cudaCommon_allocateArrayToDevice(&referenceMaskCuda, activeVoxelNumber);
    NR_CUDA_SAFE_CALL(cudaMemcpy(referenceMaskCuda, targetMask, activeVoxelNumber * sizeof(*targetMask), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaFreeHost(targetMask));
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
    NR_CUDA_SAFE_CALL(cudaMalloc(&transformationMatrixCuda, sizeof(mat44)));
    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrixCuda, transformationMatrixCptr, sizeof(mat44), cudaMemcpyHostToDevice));
    free(transformationMatrixCptr);
}
/* *************************************************************** */
nifti_image* CudaContent::GetWarped() {
    DownloadImage(warped, warpedCuda, warped->datatype);
    return warped;
}
/* *************************************************************** */
void CudaContent::SetWarped(nifti_image *warpedIn) {
    Content::SetWarped(warpedIn);
    DeallocateWarped();
    if (!warped) return;

    reg_tools_changeDatatype<float>(warped);
    AllocateWarped();
    cudaCommon_transferNiftiToArrayOnDevice(warpedCuda, warped);
}
/* *************************************************************** */
void CudaContent::UpdateWarped() {
    cudaCommon_transferNiftiToArrayOnDevice(warpedCuda, warped);
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
        return static_cast<unsigned>(intensity > 0 ? reg_round(intensity) : 0);
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
    image->data = malloc(size * image->nbyper);
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
        FillImageData<unsigned>(image, memoryObject, datatype);
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
