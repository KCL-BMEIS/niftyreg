#ifndef CUDARESAMPLEIMAGEKERNEL_H
#define CUDARESAMPLEIMAGEKERNEL_H

#include "ResampleImageKernel.h"
#include "CUDAAladinContent.h"

/*
 * kernel functions for image resampling with three interpolation variations
 * */
class CUDAResampleImageKernel: public ResampleImageKernel {
public:
    CUDAResampleImageKernel(AladinContent *conIn, std::string name);
    void calculate(int interp,
                        float paddingValue,
                        bool *dti_timepoint = NULL,
                        mat33 *jacMat = NULL);
private:
    nifti_image *floatingImage;
    nifti_image *warpedImage;

    //cuda ptrs
    float* floatingImageArray_d;
    float* floIJKMat_d;
    float* warpedImageArray_d;
    float* deformationFieldImageArray_d;
    int *mask_d;

    //CUDAContextSingletton *cudaSContext;
    CudaAladinContent *con;
};

#endif // CUDARESAMPLEIMAGEKERNEL_H
