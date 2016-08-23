#include "CPUBlockMatchingKernel.h"

CPUBlockMatchingKernel::CPUBlockMatchingKernel(GlobalContent *conIn, std::string name) : BlockMatchingKernel(name) {
    //cast to the "real type"
    con = dynamic_cast<AladinContent*>(conIn);
    reference = con->getCurrentReference();
    warped = con->getCurrentWarped();
    params = con->getBlockMatchingParams();
    mask = con->getCurrentReferenceMask();
}

void CPUBlockMatchingKernel::calculate() {
    block_matching_method(this->reference, this->warped, this->params, this->mask);
}
//
