#ifndef CLGLOBALCONTENT_H
#define CLGLOBALCONTENT_H


#include "GlobalContent.h"
#include "CLContextSingletton.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class ClGlobalContent: public virtual GlobalContent {

public:
    //constructors
    ClGlobalContent(int refTimePoint,int floTimePoint);
    virtual ~ClGlobalContent();

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();

    //opencl getters
    cl_mem getReferenceImageArrayClmem();
    cl_mem getFloatingImageArrayClmem();
    cl_mem getWarpedImageClmem();
    cl_mem getDeformationFieldArrayClmem();
    cl_mem getMaskClmem();
    cl_mem getRefMatClmem();
    cl_mem getFloMatClmem();
    int *getReferenceDims();
    int *getFloatingDims();

    //cpu getters with data downloaded from device
    nifti_image* getCurrentWarped(int datatype);
    nifti_image* getCurrentDeformationField();

    //setters
    virtual void setCurrentReference(nifti_image* currentRefIn);
    virtual void setCurrentReferenceMask(int *maskIn, size_t size);
    virtual void setCurrentFloating(nifti_image* currentFloIn);
    virtual void setCurrentWarped(nifti_image *warpedImageIn);
    virtual void setCurrentDeformationField(nifti_image *currentDeformationFieldIn);
    //
    virtual bool isCurrentComputationDoubleCapable();

protected:
    CLContextSingletton *sContext;
    cl_context clContext;
    cl_int errNum;
    cl_command_queue commandQueue;

    cl_mem referenceImageClmem;
    cl_mem floatingImageClmem;
    cl_mem warpedImageClmem;
    cl_mem deformationFieldClmem;
    cl_mem maskClmem;
    cl_mem refMatClmem;
    cl_mem floMatClmem;

    int referenceDims[4];
    int floatingDims[4];

    void downloadImage(nifti_image *image,
                       cl_mem memoryObject,
                       int datatype);
    template<class T>
    void fillImageData(nifti_image *image,
                       cl_mem memoryObject,
                       int type);
    template<class T>
    T fillWarpedImageData(float intensity,
                          int datatype);

private:
    void freeClPtrs();

};

#endif // CLGLOBALCONTENT_H
