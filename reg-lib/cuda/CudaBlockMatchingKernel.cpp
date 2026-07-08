#include "CudaBlockMatchingKernel.h"
#include "blockMatchingKernel.h"

/* *************************************************************** */
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Content *conIn) : BlockMatchingKernel() {
    //get CudaAladinContent ptr
    CudaAladinContent *con = dynamic_cast<CudaAladinContent*>(conIn);

    //get cpu ptrs
    reference = con->AladinContent::GetReference();
    params = con->AladinContent::GetBlockMatchingParams();

    //get cuda ptrs
    referenceCuda = con->GetReferenceCuda();
    warpedCuda = con->GetWarpedCuda();
    referencePositionCuda = con->GetReferencePositionCuda();
    warpedPositionCuda = con->GetWarpedPositionCuda();
    totalBlockCuda = con->GetTotalBlockCuda();
    maskCuda = con->GetMaskCuda();
    referenceMatCuda = con->GetReferenceMatCuda();
}
/* *************************************************************** */
void CudaBlockMatchingKernel::Calculate() {
    block_matching_method_gpu(reference,
                              params,
                              referenceCuda,
                              warpedCuda,
                              referencePositionCuda,
                              warpedPositionCuda,
                              totalBlockCuda,
                              maskCuda,
                              referenceMatCuda);
}
/* *************************************************************** */
