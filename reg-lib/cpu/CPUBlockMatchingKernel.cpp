#include "CPUBlockMatchingKernel.h"

CPUBlockMatchingKernel::CPUBlockMatchingKernel(AladinContent *con, std::string name) : BlockMatchingKernel(name) {
    reference = con->getCurrentReference();
    warped = con->getCurrentWarped();
    params = con->getBlockMatchingParams();
    mask = con->getCurrentReferenceMask();
}

void CPUBlockMatchingKernel::calculate() {
    block_matching_method(this->reference, this->warped, this->params, this->mask);
}
//
