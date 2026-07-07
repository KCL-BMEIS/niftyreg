#include "CudaAffineDeformationFieldKernel.h"
#include "affineDeformationKernel.h"

/* *************************************************************** */
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Content *conIn) : AffineDeformationFieldKernel() {
   CudaAladinContent *con = dynamic_cast<CudaAladinContent*>(conIn);

   //get necessary cpu ptrs
   this->deformationFieldImage = con->AladinContent::GetDeformationField();
   this->affineTransformation = con->AladinContent::GetTransformationMatrix();

   //get necessary cuda ptrs
   maskCuda = con->GetMaskCuda();
   deformationFieldCuda = con->GetDeformationFieldCuda();
   transformationMatrixCuda = con->GetTransformationMatrixCuda();
}
/* *************************************************************** */
void CudaAffineDeformationFieldKernel::Calculate(bool compose) {
   launchAffine(this->affineTransformation,
                this->deformationFieldImage,
                deformationFieldCuda,
                maskCuda,
                transformationMatrixCuda,
                compose);
}
/* *************************************************************** */
