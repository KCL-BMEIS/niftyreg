#include "CpuBlockMatchingKernel.h"

/* *************************************************************** */
CpuBlockMatchingKernel::CpuBlockMatchingKernel(Content *conIn) : BlockMatchingKernel() {
    AladinContent *con = static_cast<AladinContent*>(conIn);
    reference = con->GetReference();
    warped = con->GetWarped();
    params = con->GetBlockMatchingParams();
    mask = con->GetReferenceMask();
}
/* *************************************************************** */
void CpuBlockMatchingKernel::Calculate() {
    block_matching_method(reference, warped, params, mask);
}
/* *************************************************************** */
