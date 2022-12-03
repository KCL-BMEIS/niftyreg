#pragma once

#include "AladinContent.h"
#include "CudaContextSingleton.h"
#include "_reg_tools.h"

class CudaAladinContent: public AladinContent {
public:
    CudaAladinContent(nifti_image *referenceIn,
                      nifti_image *floatingIn,
                      int *referenceMaskIn = nullptr,
                      mat44 *transformationMatrixIn = nullptr,
                      size_t bytesIn = sizeof(float),
                      const unsigned int percentageOfBlocks = 0,
                      const unsigned int inlierLts = 0,
                      int blockStepSize = 0);
    ~CudaAladinContent();

    bool IsCurrentComputationDoubleCapable() override;

    // Device getters
    float* GetReferenceImageArray_d();
    float* GetFloatingImageArray_d();
    float* GetWarpedImageArray_d();
    float* GetTransformationMatrix_d();
    float* GetReferencePosition_d();
    float* GetWarpedPosition_d();
    float* GetDeformationFieldArray_d();
    float* GetReferenceMat_d();
    float* GetFloIJKMat_d();

    //	float* GetAR_d(); // Removed until CUDA SVD is added back
    //	float* GetU_d(); // Removed until CUDA SVD is added back
    //	float* GetVT_d(); // Removed until CUDA SVD is added back
    //	float* GetSigma_d(); // Removed until CUDA SVD is added back
    //	float* GetLengths_d(); // Removed until CUDA SVD is added back
    //	float* GetNewWarpedPos_d(); // Removed until CUDA SVD is added back

    int* GetTotalBlock_d();
    int* GetMask_d();

    int* GetReferenceDims();
    int* GetFloatingDims();

    // CPU getters with data downloaded from device
    _reg_blockMatchingParam* GetBlockMatchingParams() override;
    nifti_image* GetDeformationField() override;
    nifti_image* GetWarped(int datatype, int index = 0) override;

    // Setters
    void SetTransformationMatrix(mat44 *transformationMatrixIn) override;
    void SetWarped(nifti_image *warpedImageIn) override;
    void SetDeformationField(nifti_image *deformationFieldIn) override;
    void SetReferenceMask(int *referenceMaskIn) override;
    void SetBlockMatchingParams(_reg_blockMatchingParam* bmp) override;

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

    //svd
    //	float *AR_d;//A and then pseudoinverse  // Removed until CUDA SVD is added back
    //	float *U_d; // Removed until CUDA SVD is added back
    //	float *VT_d; // Removed until CUDA SVD is added back
    //	float *Sigma_d; // Removed until CUDA SVD is added back
    //	float *lengths_d; // Removed until CUDA SVD is added back
    //	float *newWarpedPos_d; // Removed until CUDA SVD is added back

    int referenceDims[4];
    int floatingDims[4];

    void DownloadImage(nifti_image *image, float* memoryObject, int datatype);
    template<class T>
    void FillImageData(nifti_image *image, float* memoryObject, int type);

    template<class FloatingTYPE>
    FloatingTYPE FillWarpedImageData(float intensity, int datatype);
};
