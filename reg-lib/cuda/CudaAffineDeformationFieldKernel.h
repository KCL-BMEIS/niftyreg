#pragma once

#include "AffineDeformationFieldKernel.h"
#include "CudaAladinContent.h"

//Kernel functions for affine deformation field
class CudaAffineDeformationFieldKernel: public AffineDeformationFieldKernel {
public:
    CudaAffineDeformationFieldKernel(Content *conIn);
    virtual void Calculate(bool compose = false) override;

private:
    mat44 *affineTransformation;
    nifti_image *deformationFieldImage;

    float *deformationFieldArray_d, *transformationMatrix_d;
    int *mask_d;
};
