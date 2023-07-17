#include "ClLtsKernel.h"

/* *************************************************************** */
ClLtsKernel::ClLtsKernel(Content *conIn) : LtsKernel() {
    //populate the ClAladinContent object ptr
    ClAladinContent *con = static_cast<ClAladinContent*>(conIn);

    //get necessary cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void ClLtsKernel::Calculate(bool affine) {
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
