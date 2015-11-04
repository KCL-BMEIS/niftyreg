#ifndef CUDAAFFINEDEFORMATIONFIELDKERNEL_H
#define CUDAAFFINEDEFORMATIONFIELDKERNEL_H

#include "AffineDeformationFieldKernel.h"
#include "CUDAAladinContent.h"

//Kernel functions for affine deformation field
class CUDAAffineDeformationFieldKernel: public AffineDeformationFieldKernel
{
public:
    CUDAAffineDeformationFieldKernel(AladinContent *conIn, std::string nameIn);
    void calculate(bool compose = false);
private:
    mat44 *affineTransformation;
    nifti_image *deformationFieldImage;

    float *deformationFieldArray_d, *transformationMatrix_d;
    int *mask_d;

    CudaAladinContent *con;

    //CUDAContextSingletton *cudaSContext;
    //CUContext cudaContext;
};

#endif // CUDAAFFINEDEFORMATIONFIELDKERNEL_H
