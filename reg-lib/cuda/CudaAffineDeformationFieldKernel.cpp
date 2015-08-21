#include "CudaAffineDeformationFieldKernel.h"

/* *************************************************************** */
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Content *conIn, std::string nameIn) :
        AffineDeformationFieldKernel(nameIn)
{
    con = static_cast<CudaContent*>(conIn);

    //get necessary cpu ptrs
    this->deformationFieldImage = con->Content::getCurrentDeformationField();
    this->affineTransformation = con->Content::getTransformationMatrix();

    //get necessary cuda ptrs
    mask_d = con->getMask_d();
    deformationFieldArray_d = con->getDeformationFieldArray_d();
    transformationMatrix_d = con->getTransformationMatrix_d();
}
/* *************************************************************** */
void CudaAffineDeformationFieldKernel::calculate(bool compose)
{
    launchAffine(this->affineTransformation,
                     this->deformationFieldImage,
                     &deformationFieldArray_d,
                     &mask_d,
                     &transformationMatrix_d,
                     compose);
}
/* *************************************************************** */
