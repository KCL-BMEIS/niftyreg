#pragma once

#include "AladinContent.h"
#include "CudaContextSingleton.h"

#include "_reg_tools.h"

class CudaAladinContent: public AladinContent {
public:
    CudaAladinContent();
    CudaAladinContent(nifti_image *currentReferenceIn,
                      nifti_image *currentFloatingIn,
                      int *currentReferenceMaskIn,
                      size_t byte,
                      const unsigned int blockPercentage,
                      const unsigned int inlierLts,
                      int blockStep);
    CudaAladinContent(nifti_image *currentReferenceIn,
                      nifti_image *currentFloatingIn,
                      int *currentReferenceMaskIn,
                      size_t byte);
    CudaAladinContent(nifti_image *currentReferenceIn,
                      nifti_image *currentFloatingIn,
                      int *currentReferenceMaskIn,
                      mat44 *transMat,
                      size_t byte,
                      const unsigned int blockPercentage,
                      const unsigned int inlierLts,
                      int blockStep);
    CudaAladinContent(nifti_image *currentReferenceIn,
                      nifti_image *currentFloatingIn,
                      int *currentReferenceMaskIn,
                      mat44 *transMat,
                      size_t byte);
    ~CudaAladinContent();

    bool IsCurrentComputationDoubleCapable();

    //device getters
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

    //cpu getters and setters
    _reg_blockMatchingParam* GetBlockMatchingParams();
    nifti_image* GetCurrentDeformationField();
    nifti_image* GetCurrentWarped(int typ);

    void SetTransformationMatrix(mat44 *transformationMatrixIn);
    void SetCurrentWarped(nifti_image *warpedImageIn);
    void SetCurrentDeformationField(nifti_image *currentDeformationFieldIn);
    void SetCurrentReferenceMask(int *maskIn, size_t size);
    void SetBlockMatchingParams(_reg_blockMatchingParam* bmp);

private:
    void InitVars();

    void AllocateCuPtrs();
    void FreeCuPtrs();

    CudaContextSingleton *cudaSContext;
    CUcontext cudaContext;

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
