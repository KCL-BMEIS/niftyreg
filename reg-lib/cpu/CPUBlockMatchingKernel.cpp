#include "CPUBlockMatchingKernel.h"

void CPUBlockMatchingKernel::calculate() {
    block_matching_method(this->reference, this->warped, this->params, this->mask);
}
//
