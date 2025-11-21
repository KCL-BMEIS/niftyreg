#include <cuda_runtime.h>
#include <cuda.h>
#include "CudaLtsKernel.h"

/* *************************************************************** */
CudaLtsKernel::CudaLtsKernel(Content *conIn) : LtsKernel() {
    //get CudaAladinContent ptr
    con = static_cast<CudaAladinContent*>(conIn);

    //get cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void CudaLtsKernel::Calculate(bool affine) {
    blockMatchingParams = con->GetBlockMatchingParams();
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
