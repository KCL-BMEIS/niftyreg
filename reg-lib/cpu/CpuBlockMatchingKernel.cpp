#include "CpuBlockMatchingKernel.h"

CpuBlockMatchingKernel::CpuBlockMatchingKernel(AladinContent *con, std::string name) : BlockMatchingKernel(name) {
    reference = con->GetCurrentReference();
    warped = con->GetCurrentWarped();
    params = con->GetBlockMatchingParams();
    mask = con->GetCurrentReferenceMask();
}

void CpuBlockMatchingKernel::Calculate() {
    block_matching_method(this->reference, this->warped, this->params, this->mask);
}
//
