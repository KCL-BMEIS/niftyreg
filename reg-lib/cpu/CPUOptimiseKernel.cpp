#include "CPUOptimiseKernel.h"

CPUOptimiseKernel::CPUOptimiseKernel(GlobalContent *conIn, std::string name) : OptimiseKernel(name) {
    //cast to the "real type"
    con = dynamic_cast<AladinContent*>(conIn);
    transformationMatrix = con->getTransformationMatrix();
    blockMatchingParams = con->getBlockMatchingParams();
}

void CPUOptimiseKernel::calculate(bool affine) {
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
