#include "ClOptimiseKernel.h"

/* *************************************************************** */
ClOptimiseKernel::ClOptimiseKernel(Content *conIn) : OptimiseKernel() {
    //populate the CLAladinContent object ptr
    ClAladinContent *con = static_cast<ClAladinContent*>(conIn);

    //get necessary cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void ClOptimiseKernel::Calculate(bool affine) {
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
