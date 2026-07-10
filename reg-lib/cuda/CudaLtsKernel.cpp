#include <cuda_runtime.h>
#include <cuda.h>
#include "CudaLtsKernel.h"
#include "CudaLts.hpp"

/* *************************************************************** */
CudaLtsKernel::CudaLtsKernel(Content *conIn) : LtsKernel() {
    //get CudaAladinContent ptr
    con = dynamic_cast<CudaAladinContent*>(conIn);

    //get cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void CudaLtsKernel::Calculate(bool affine) {
    Cuda::OptimizeLts(blockMatchingParams, transformationMatrix, con->GetReferencePositionCuda(), con->GetWarpedPositionCuda(), affine);
}
/* *************************************************************** */
