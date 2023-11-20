#include "CudaContent.h"

/* *************************************************************** */
CudaContent::CudaContent(nifti_image *referenceIn,
                         nifti_image *floatingIn,
                         int *referenceMaskIn,
                         mat44 *transformationMatrixIn,
                         size_t bytesIn):
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateReference();
    AllocateFloating();
    AllocateWarped();
    AllocateDeformationField();
    SetReferenceMask(referenceMask);
    SetTransformationMatrix(transformationMatrix);
}
/* *************************************************************** */
CudaContent::~CudaContent() {
    DeallocateWarped();
    DeallocateDeformationField();
    SetReferenceMask(nullptr);
    SetTransformationMatrix(nullptr);
}
/* *************************************************************** */
void CudaContent::AllocateReference() {
    if (reference->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(reference);
    Cuda::Allocate(&referenceCuda, reference->nvox);
    referenceCudaManaged.reset(referenceCuda);
    Cuda::TransferNiftiToDevice(referenceCuda, reference);
}
/* *************************************************************** */
void CudaContent::AllocateFloating() {
    if (floating->nbyper != NIFTI_TYPE_FLOAT32)
        reg_tools_changeDatatype<float>(floating);
    Cuda::Allocate(&floatingCuda, floating->nvox);
    floatingCudaManaged.reset(floatingCuda);
    Cuda::TransferNiftiToDevice(floatingCuda, floating);
}
/* *************************************************************** */
void CudaContent::AllocateDeformationField() {
    Cuda::Allocate(&deformationFieldCuda, deformationField->dim);
    UpdateDeformationField();
}
/* *************************************************************** */
void CudaContent::DeallocateDeformationField() {
    if (deformationFieldCuda) {
        Cuda::Free(deformationFieldCuda);
        deformationFieldCuda = nullptr;
    }
}
/* *************************************************************** */
void CudaContent::AllocateWarped() {
    Cuda::Allocate(&warpedCuda, warped->nvox);
}
/* *************************************************************** */
void CudaContent::DeallocateWarped() {
    if (warpedCuda) {
        Cuda::Free(warpedCuda);
        warpedCuda = nullptr;
    }
}
/* *************************************************************** */
bool CudaContent::IsCurrentComputationDoubleCapable() {
    return CudaContext::GetInstance().IsCardDoubleCapable();
}
/* *************************************************************** */
nifti_image* CudaContent::GetDeformationField() {
    Cuda::TransferFromDeviceToNifti(deformationField, deformationFieldCuda);
    return deformationField;
}
/* *************************************************************** */
void CudaContent::SetDeformationField(nifti_image *deformationFieldIn) {
    Content::SetDeformationField(deformationFieldIn);
    DeallocateDeformationField();
    if (!deformationField) return;

    AllocateDeformationField();
    Cuda::TransferNiftiToDevice(deformationFieldCuda, deformationField);
}
/* *************************************************************** */
void CudaContent::UpdateDeformationField() {
    Cuda::TransferNiftiToDevice(deformationFieldCuda, deformationField);
}
/* *************************************************************** */
void CudaContent::SetReferenceMask(int *referenceMaskIn) {
    Content::SetReferenceMask(referenceMaskIn);

    if (referenceMaskCuda) {
        Cuda::Free(referenceMaskCuda);
        referenceMaskCuda = nullptr;
    }

    activeVoxelNumber = 0;
    if (!referenceMask) return;

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(reference, 3);
    thrust::host_vector<int> mask(voxelNumber);
    int *maskPtr = mask.data();
    for (size_t i = 0; i < voxelNumber; i++) {
        if (referenceMask[i] != -1) {
            *maskPtr++ = static_cast<int>(i);
            activeVoxelNumber++;
        }
    }

    Cuda::Allocate(&referenceMaskCuda, activeVoxelNumber);
    thrust::copy_n(mask.begin(), activeVoxelNumber, thrust::device_ptr<int>(referenceMaskCuda));
}
/* *************************************************************** */
void CudaContent::SetTransformationMatrix(mat44 *transformationMatrixIn) {
    Content::SetTransformationMatrix(transformationMatrixIn);

    if (transformationMatrixCuda) {
        Cuda::Free(transformationMatrixCuda);
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
    Cuda::TransferNiftiToDevice(warpedCuda, warped);
}
/* *************************************************************** */
void CudaContent::UpdateWarped() {
    Cuda::TransferNiftiToDevice(warpedCuda, warped);
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
        intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
        return static_cast<unsigned char>(intensity > 0 ? Round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT16:
        intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
        return static_cast<unsigned short>(intensity > 0 ? Round(intensity) : 0);
        break;
    case NIFTI_TYPE_UINT32:
        intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
        return static_cast<unsigned>(intensity > 0 ? Round(intensity) : 0);
        break;
    default:
        return static_cast<DataType>(Round(intensity));
        break;
    }
}
/* *************************************************************** */
template<class DataType>
void CudaContent::FillImageData(nifti_image *image, float *memoryObject, int datatype) {
    size_t size = image->nvox;
    float *buffer = (float*)malloc(size * sizeof(float));

    Cuda::TransferFromDeviceToHost(buffer, memoryObject, size);

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
        NR_FATAL_ERROR("Unsupported type");
    }
}
/* *************************************************************** */
