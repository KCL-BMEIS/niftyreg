#include "CudaContent.h"

/* *************************************************************** */
CudaContent::CudaContent(NiftiImage& referenceIn,
                         NiftiImage& floatingIn,
                         int *referenceMaskIn,
                         mat44 *transformationMatrixIn,
                         size_t bytesIn):
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, sizeof(float)) {
    AllocateReference();
    AllocateFloating();
    AllocateWarped();
    AllocateDeformationField();
    CudaContent::SetReferenceMask(referenceMask);
    CudaContent::SetTransformationMatrix(transformationMatrix);
}
/* *************************************************************** */
CudaContent::~CudaContent() {
    DeallocateWarped();
    DeallocateDeformationField();
    CudaContent::SetReferenceMask(nullptr);
    CudaContent::SetTransformationMatrix(nullptr);
}
/* *************************************************************** */
void CudaContent::AllocateReference() {
    if (reference->nbyper != NIFTI_TYPE_FLOAT32)
        reference.changeDatatype(NIFTI_TYPE_FLOAT32);
    Cuda::Allocate(&referenceCuda, reference->nvox);
    referenceCudaManaged.reset(referenceCuda);
    Cuda::TransferNiftiToDevice(referenceCuda, reference);
}
/* *************************************************************** */
void CudaContent::AllocateFloating() {
    if (floating->nbyper != NIFTI_TYPE_FLOAT32)
        floating.changeDatatype(NIFTI_TYPE_FLOAT32);
    Cuda::Allocate(&floatingCuda, floating->nvox);
    floatingCudaManaged.reset(floatingCuda);
    Cuda::TransferNiftiToDevice(floatingCuda, floating);
}
/* *************************************************************** */
void CudaContent::AllocateDeformationField() {
    Cuda::Allocate(&deformationFieldCuda, deformationField->dim);
    CudaContent::UpdateDeformationField();
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
NiftiImage& CudaContent::GetDeformationField() {
    Cuda::TransferFromDeviceToNifti(deformationField, deformationFieldCuda);
    return deformationField;
}
/* *************************************************************** */
void CudaContent::SetDeformationField(NiftiImage&& deformationFieldIn) {
    Content::SetDeformationField(std::move(deformationFieldIn));
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

    const size_t voxelNumber = reference.nVoxelsPerVolume();
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
NiftiImage& CudaContent::GetWarped() {
    DownloadImage(warped, warpedCuda, warped->datatype);
    return warped;
}
/* *************************************************************** */
void CudaContent::SetWarped(NiftiImage&& warpedIn) {
    Content::SetWarped(std::move(warpedIn));
    DeallocateWarped();
    if (!warped) return;

    if (warped->nbyper != NIFTI_TYPE_FLOAT32)
        warped.changeDatatype(NIFTI_TYPE_FLOAT32);

    AllocateWarped();
    Cuda::TransferNiftiToDevice(warpedCuda, warped);
}
/* *************************************************************** */
void CudaContent::UpdateWarped() {
    Cuda::TransferNiftiToDevice(warpedCuda, warped);
}
/* *************************************************************** */
void CudaContent::DownloadImage(NiftiImage& image, float *memoryObject, int datatype) {
    const size_t size = image->nvox;
    unique_ptr<float[]> buffer(new float[size]);

    Cuda::TransferFromDeviceToHost(buffer.get(), memoryObject, size);

    std::visit([&](auto&& dataType) {
        using DataType = std::decay_t<decltype(dataType)>;
        image->datatype = datatype;
        image->nbyper = sizeof(DataType);
        image.realloc();
        DataType *data = static_cast<DataType*>(image->data);
        for (size_t i = 0; i < size; ++i)
            data[i] = static_cast<DataType>(image.clampData(buffer[i]));
    }, image.getDataType());
}
/* *************************************************************** */
