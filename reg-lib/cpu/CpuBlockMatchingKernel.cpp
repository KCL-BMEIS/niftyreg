#include "CpuBlockMatchingKernel.h"

/* *************************************************************** */
CpuBlockMatchingKernel::CpuBlockMatchingKernel(Content *conIn) : BlockMatchingKernel() {
    AladinContent *con = static_cast<AladinContent*>(conIn);
    reference = con->GetCurrentReference();
    warped = con->GetCurrentWarped();
    params = con->GetBlockMatchingParams();
    mask = con->GetCurrentReferenceMask();
}
/* *************************************************************** */
void CpuBlockMatchingKernel::Calculate() {
    block_matching_method(reference, warped, params, mask);
}
/* *************************************************************** */
