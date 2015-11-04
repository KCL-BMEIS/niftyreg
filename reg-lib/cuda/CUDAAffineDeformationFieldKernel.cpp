#include "CUDAAffineDeformationFieldKernel.h"
#include "affineDeformationKernel.h"

/* *************************************************************** */
CUDAAffineDeformationFieldKernel::CUDAAffineDeformationFieldKernel(AladinContent *conIn, std::string nameIn) :
   AffineDeformationFieldKernel(nameIn)
{
   con = static_cast<CudaAladinContent*>(conIn);

   //get necessary cpu ptrs
   this->deformationFieldImage = con->AladinContent::getCurrentDeformationField();
   this->affineTransformation = con->AladinContent::getTransformationMatrix();

   //get necessary cuda ptrs
   mask_d = con->getMask_d();
   deformationFieldArray_d = con->getDeformationFieldArray_d();
   transformationMatrix_d = con->getTransformationMatrix_d();

   //cudaSContext = &CUDAContextSingletton::Instance();
   //cudaContext = cudaSContext->getContext();
}
/* *************************************************************** */
void CUDAAffineDeformationFieldKernel::calculate(bool compose)
{
   launchAffine(this->affineTransformation,
                this->deformationFieldImage,
                &deformationFieldArray_d,
                &mask_d,
                &transformationMatrix_d,
                compose);
}
/* *************************************************************** */
