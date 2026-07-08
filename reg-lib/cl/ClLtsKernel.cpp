#include "ClLtsKernel.h"

/* *************************************************************** */
ClLtsKernel::ClLtsKernel(Content *conIn) : LtsKernel() {
    //populate the ClAladinContent object ptr
    con = dynamic_cast<ClAladinContent*>(conIn);

    //get necessary cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void ClLtsKernel::Calculate(bool affine) {
    blockMatchingParams = con->GetBlockMatchingParams();
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
