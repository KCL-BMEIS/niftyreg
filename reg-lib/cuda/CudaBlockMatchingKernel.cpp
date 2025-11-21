#include "CudaBlockMatchingKernel.h"
#include "blockMatchingKernel.h"

/* *************************************************************** */
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Content *conIn) : BlockMatchingKernel() {
    //get CudaAladinContent ptr
    CudaAladinContent *con = static_cast<CudaAladinContent*>(conIn);

    //get cpu ptrs
    reference = con->AladinContent::GetReference();
    params = con->AladinContent::GetBlockMatchingParams();

    //get cuda ptrs
    referenceImageArray_d = con->GetReferenceImageArray_d();
    warpedImageArray_d = con->GetWarpedImageArray_d();
    referencePosition_d = con->GetReferencePosition_d();
    warpedPosition_d = con->GetWarpedPosition_d();
    totalBlock_d = con->GetTotalBlock_d();
    mask_d = con->GetMask_d();
    referenceMat_d = con->GetReferenceMat_d();
}
/* *************************************************************** */
void CudaBlockMatchingKernel::Calculate() {
    block_matching_method_gpu(reference,
                              params,
                              referenceImageArray_d,
                              warpedImageArray_d,
                              referencePosition_d,
                              warpedPosition_d,
                              totalBlock_d,
                              mask_d,
                              referenceMat_d);
}
/* *************************************************************** */
