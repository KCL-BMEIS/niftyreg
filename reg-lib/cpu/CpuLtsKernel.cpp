#include "CpuLtsKernel.h"

/* *************************************************************** */
CpuLtsKernel::CpuLtsKernel(Content *conIn) : LtsKernel() {
    AladinContent *con = static_cast<AladinContent*>(conIn);
    transformationMatrix = con->GetTransformationMatrix();
    blockMatchingParams = con->GetBlockMatchingParams();
}
/* *************************************************************** */
void CpuLtsKernel::Calculate(bool affine) {
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
