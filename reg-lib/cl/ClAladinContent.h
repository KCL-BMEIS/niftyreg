#pragma once

#include "AladinContent.h"
#include "ClContextSingleton.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class ClAladinContent: public AladinContent {
public:
    //constructors
    ClAladinContent();
    ClAladinContent(nifti_image *currentReferenceIn,
                    nifti_image *currentFloatingIn,
                    int *currentReferenceMaskIn,
                    size_t byte,
                    const unsigned int blockPercentage,
                    const unsigned int inlierLts,
                    int blockStep);
    ClAladinContent(nifti_image *currentReferenceIn,
                    nifti_image *currentFloatingIn,
                    int *currentReferenceMaskIn,
                    size_t byte);
    ClAladinContent(nifti_image *currentReferenceIn,
                    nifti_image *currentFloatingIn,
                    int *currentReferenceMaskIn,
                    mat44 *transMat,
                    size_t byte,
                    const unsigned int blockPercentage,
                    const unsigned int inlierLts,
                    int blockStep);
    ClAladinContent(nifti_image *currentReferenceIn,
                    nifti_image *currentFloatingIn,
                    int *currentReferenceMaskIn,
                    mat44 *transMat,
                    size_t byte);
    ~ClAladinContent();

    bool IsCurrentComputationDoubleCapable();

    //opencl getters
    cl_mem GetReferenceImageArrayClmem();
    cl_mem GetFloatingImageArrayClmem();
    cl_mem GetWarpedImageClmem();
    cl_mem GetReferencePositionClmem();
    cl_mem GetWarpedPositionClmem();
    cl_mem GetDeformationFieldArrayClmem();
    cl_mem GetTotalBlockClmem();
    cl_mem GetMaskClmem();
    cl_mem GetRefMatClmem();
    cl_mem GetFloMatClmem();
    int* GetReferenceDims();
    int* GetFloatingDims();

    //cpu getters with data downloaded from device
    _reg_blockMatchingParam* GetBlockMatchingParams();
    nifti_image* GetCurrentDeformationField();
    nifti_image* GetCurrentWarped(int typ);

    //setters
    void SetTransformationMatrix(mat44 *transformationMatrixIn);
    void SetCurrentWarped(nifti_image *warpedImageIn);
    void SetCurrentDeformationField(nifti_image *currentDeformationFieldIn);
    void SetCurrentReferenceMask(int *maskIn, size_t size);
    void SetBlockMatchingParams(_reg_blockMatchingParam* bmp);


private:
    void InitVars();
    void AllocateClPtrs();
    void FreeClPtrs();

    ClContextSingleton *sContext;
    cl_context clContext;
    cl_int errNum;
    cl_command_queue commandQueue;

    cl_mem referenceImageClmem;
    cl_mem floatingImageClmem;
    cl_mem warpedImageClmem;
    cl_mem deformationFieldClmem;
    cl_mem referencePositionClmem;
    cl_mem warpedPositionClmem;
    cl_mem totalBlockClmem;
    cl_mem maskClmem;
    cl_mem refMatClmem;
    cl_mem floMatClmem;

    int referenceDims[4];
    int floatingDims[4];

    unsigned int nVoxels;

    void DownloadImage(nifti_image *image, cl_mem memoryObject, int datatype);
    template<class T>
    void FillImageData(nifti_image *image, cl_mem memoryObject, int type);
    template<class T>
    T FillWarpedImageData(float intensity, int datatype);
};
