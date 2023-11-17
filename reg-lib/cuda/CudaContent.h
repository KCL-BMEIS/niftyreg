#pragma once

#include "Content.h"
#include "CudaCommon.hpp"

class CudaContent: public virtual Content {
public:
    CudaContent() = delete;
    CudaContent(nifti_image *referenceIn,
                nifti_image *floatingIn,
                int *referenceMaskIn = nullptr,
                mat44 *transformationMatrixIn = nullptr,
                size_t bytesIn = sizeof(float));
    virtual ~CudaContent();

    virtual bool IsCurrentComputationDoubleCapable() override;

    // Getters
    virtual nifti_image* GetDeformationField() override;
    virtual nifti_image* GetWarped() override;
    virtual float* GetReferenceCuda() { return referenceCuda; }
    virtual float* GetFloatingCuda() { return floatingCuda; }
    virtual float4* GetDeformationFieldCuda() { return deformationFieldCuda; }
    virtual int* GetReferenceMaskCuda() { return referenceMaskCuda; }
    virtual float* GetTransformationMatrixCuda() { return transformationMatrixCuda; }
    virtual float* GetWarpedCuda() { return warpedCuda; }

    // Methods for transferring data from nifti to device
    virtual void UpdateDeformationField() override;
    virtual void UpdateWarped() override;

protected:
    float *referenceCuda = nullptr;
    Cuda::UniquePtr<float> referenceCudaManaged;
    float *floatingCuda = nullptr;
    Cuda::UniquePtr<float> floatingCudaManaged;
    float4 *deformationFieldCuda = nullptr;
    int *referenceMaskCuda = nullptr;
    float *transformationMatrixCuda = nullptr;
    float *warpedCuda = nullptr;

private:
    void AllocateReference();
    void AllocateFloating();
    void AllocateDeformationField();
    void DeallocateDeformationField();
    void AllocateWarped();
    void DeallocateWarped();
    template<class DataType> DataType CastImageData(float intensity, int datatype);
    template<class DataType> void FillImageData(nifti_image *image, float *memoryObject, int datatype);
    void DownloadImage(nifti_image *image, float *memoryObject, int datatype);
    void SetReferenceCuda(float *referenceCudaIn) { referenceCudaManaged = nullptr; referenceCuda = referenceCudaIn; }
    void SetFloatingCuda(float *floatingCudaIn) { floatingCudaManaged = nullptr; floatingCuda = floatingCudaIn; }

    // Friend classes
    friend class CudaF3d2ContentCreator;

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetDeformationField(nifti_image *deformationFieldIn) override;
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) override;
    virtual void SetWarped(nifti_image *warpedIn) override;
};
