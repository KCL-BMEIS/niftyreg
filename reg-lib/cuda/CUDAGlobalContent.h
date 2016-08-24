#ifndef CUDAGLOBALCONTENT_H
#define CUDAGLOBALCONTENT_H

#include "GlobalContent.h"
#include "CUDAContextSingletton.h"
#include "_reg_common_cuda.h"

class CudaGlobalContent : public virtual GlobalContent {

public:
        CudaGlobalContent(int refTimePoint,int floTimePoint);
        virtual ~CudaGlobalContent();

        virtual void AllocateWarped();
        virtual void ClearWarped();
        virtual void AllocateDeformationField();
        virtual void ClearDeformationField();

        //device getters
        float* getReferenceImageArray_d();
        float* getFloatingImageArray_d();
        float* getWarpedImageArray_d();
        float* getDeformationFieldArray_d();
        int *getMask_d();
        float* getReferenceMat_d();
        float* getFloIJKMat_d();
        int *getReferenceDims();
        int *getFloatingDims();
        //
        //cpu getters and setters
        nifti_image *getCurrentWarped(int typ);
        nifti_image *getCurrentDeformationField();
        //setters
        virtual void setCurrentReference(nifti_image* currentRefIn);
        virtual void setCurrentReferenceMask(int *maskIn, size_t size);
        virtual void setCurrentFloating(nifti_image* currentFloIn);
        virtual void setCurrentWarped(nifti_image *warpedImageIn);
        virtual void setCurrentDeformationField(nifti_image *currentDeformationFieldIn);
        //
        bool isCurrentComputationDoubleCapable();

protected:
        CUDAContextSingletton* cudaSContext;
        CUcontext cudaContext;

        float *referenceImageArray_d;
        float *floatingImageArray_d;
        float *warpedImageArray_d;
        float *deformationFieldArray_d;
        int *mask_d;
        float* referenceMat_d;
        float* floIJKMat_d;

        int referenceDims[4];
        int floatingDims[4];

        void downloadImage(nifti_image *image, float* memoryObject, int datatype);
        template<class T>
        void fillImageData(nifti_image *image, float* memoryObject, int type);

        template<class FloatingTYPE>
        FloatingTYPE fillWarpedImageData(float intensity, int datatype);

private:
        void freeCuPtrs();
};

#endif // CUDAGLOBALCONTENT_H
