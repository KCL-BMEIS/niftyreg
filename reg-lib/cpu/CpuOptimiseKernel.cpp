#include "CpuOptimiseKernel.h"

/* *************************************************************** */
CpuOptimiseKernel::CpuOptimiseKernel(Content *conIn) : OptimiseKernel() {
    AladinContent *con = static_cast<AladinContent*>(conIn);
    transformationMatrix = con->GetTransformationMatrix();
    blockMatchingParams = con->GetBlockMatchingParams();
}
/* *************************************************************** */
void CpuOptimiseKernel::Calculate(bool affine) {
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
