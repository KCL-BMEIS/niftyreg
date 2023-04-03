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
                    const unsigned percentageOfBlocks = 0,
                    const unsigned inlierLts = 0,
                    int blockStepSize = 0);
    virtual ~ClAladinContent();

    virtual bool IsCurrentComputationDoubleCapable() override;

    // OpenCL getters
    virtual cl_mem GetReferenceImageArrayClmem();
    virtual cl_mem GetFloatingImageArrayClmem();
    virtual cl_mem GetWarpedImageClmem();
    virtual cl_mem GetReferencePositionClmem();
    virtual cl_mem GetWarpedPositionClmem();
    virtual cl_mem GetDeformationFieldArrayClmem();
    virtual cl_mem GetTotalBlockClmem();
    virtual cl_mem GetMaskClmem();
    virtual cl_mem GetRefMatClmem();
    virtual cl_mem GetFloMatClmem();
    virtual int* GetReferenceDims();
    virtual int* GetFloatingDims();

    // CPU getters with data downloaded from device
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() override;
    virtual nifti_image* GetDeformationField() override;
    virtual nifti_image* GetWarped() override;

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

    unsigned nVoxels;

    void DownloadImage(nifti_image *image, cl_mem memoryObject, int datatype);
    template<class T>
    void FillImageData(nifti_image *image, cl_mem memoryObject, int type);
    template<class T>
    T FillWarpedImageData(float intensity, int datatype);

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) override;
    virtual void SetWarped(nifti_image *warpedImageIn) override;
    virtual void SetDeformationField(nifti_image *deformationFieldIn) override;
    virtual void SetReferenceMask(int *referenceMaskIn) override;
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam* bmp) override;
};
