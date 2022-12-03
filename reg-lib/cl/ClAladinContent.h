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
    ClAladinContent(nifti_image *referenceIn,
                    nifti_image *floatingIn,
                    int *referenceMaskIn = nullptr,
                    mat44 *transformationMatrixIn = nullptr,
                    size_t bytesIn = sizeof(float),
                    const unsigned int percentageOfBlocks = 0,
                    const unsigned int inlierLts = 0,
                    int blockStepSize = 0);
    ~ClAladinContent();

    bool IsCurrentComputationDoubleCapable() override;

    // OpenCL getters
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
