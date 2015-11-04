#include "CPUOptimiseKernel.h"

CPUOptimiseKernel::CPUOptimiseKernel(AladinContent *con, std::string name) : OptimiseKernel(name) {
    transformationMatrix = con->getTransformationMatrix();
    blockMatchingParams = con->getBlockMatchingParams();
}

void CPUOptimiseKernel::calculate(bool affine) {
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
