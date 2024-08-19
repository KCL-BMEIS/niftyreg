#pragma once

#include "AladinContent.h"
#include "CudaContext.hpp"
#include "_reg_tools.h"

class CudaAladinContent: public AladinContent {
public:
    CudaAladinContent(nifti_image *referenceIn,
                      nifti_image *floatingIn,
                      int *referenceMaskIn = nullptr,
                      mat44 *transformationMatrixIn = nullptr,
                      size_t bytesIn = sizeof(float),
                      const unsigned percentageOfBlocks = 0,
                      const unsigned inlierLts = 0,
                      int blockStepSize = 0);
    virtual ~CudaAladinContent();

    virtual bool IsCurrentComputationDoubleCapable() override;

    // Device getters
    virtual float* GetReferenceImageArray_d();
    virtual float* GetFloatingImageArray_d();
    virtual float* GetWarpedImageArray_d();
    virtual float* GetTransformationMatrix_d();
    virtual float* GetReferencePosition_d();
    virtual float* GetWarpedPosition_d();
    virtual float* GetDeformationFieldArray_d();
    virtual float* GetReferenceMat_d();
    virtual float* GetFloIJKMat_d();

    virtual int* GetTotalBlock_d();
    virtual int* GetMask_d();

    // CPU getters with data downloaded from device
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() override;
    virtual nifti_image* GetDeformationField() override;
    virtual nifti_image* GetWarped() override;

private:
    void InitVars();
    void AllocateCuPtrs();
    void FreeCuPtrs();

    float *referenceImageArray_d;
    float *floatingImageArray_d;
    float *warpedImageArray_d;
    float *deformationFieldArray_d;
    float *referencePosition_d;
    float *warpedPosition_d;
    int   *totalBlock_d, *mask_d;

    float *transformationMatrix_d;
    float *referenceMat_d;
    float *floIJKMat_d;

    template<typename DataType> void FillImageData(nifti_image *image, float *memoryObject, int datatype);
    void DownloadImage(nifti_image *image, float *memoryObject, int datatype);

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) override;
    virtual void SetWarped(nifti_image *warpedIn) override;
    virtual void SetDeformationField(nifti_image *deformationFieldIn) override;
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam *bmp) override;
};
