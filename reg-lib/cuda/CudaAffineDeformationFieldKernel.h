#ifndef CUDAAFFINEDEFORMATIONFIELDKERNEL_H
#define CUDAAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"

//Kernel functions for affine deformation field
class CudaAffineDeformationFieldKernel: public AffineDeformationFieldKernel
{
public:
    CudaAffineDeformationFieldKernel(Content *conIn, std::string nameIn);
    void calculate(bool compose = false);
private:
    mat44 *affineTransformation;
    nifti_image *deformationFieldImage;

    float *deformationFieldArray_d, *transformationMatrix_d;
    int *mask_d;
    CudaContent *con;

};

#endif // CUDAAFFINEDEFORMATIONFIELDKERNEL_H
