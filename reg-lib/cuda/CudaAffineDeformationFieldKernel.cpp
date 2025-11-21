#include "CudaAffineDeformationFieldKernel.h"
#include "affineDeformationKernel.h"

/* *************************************************************** */
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Content *conIn) : AffineDeformationFieldKernel() {
   CudaAladinContent *con = static_cast<CudaAladinContent*>(conIn);

   //get necessary cpu ptrs
   this->deformationFieldImage = con->AladinContent::GetDeformationField();
   this->affineTransformation = con->AladinContent::GetTransformationMatrix();

   //get necessary cuda ptrs
   mask_d = con->GetMask_d();
   deformationFieldArray_d = con->GetDeformationFieldArray_d();
   transformationMatrix_d = con->GetTransformationMatrix_d();
}
/* *************************************************************** */
void CudaAffineDeformationFieldKernel::Calculate(bool compose) {
   launchAffine(this->affineTransformation,
                this->deformationFieldImage,
                deformationFieldArray_d,
                mask_d,
                transformationMatrix_d,
                compose);
}
/* *************************************************************** */
