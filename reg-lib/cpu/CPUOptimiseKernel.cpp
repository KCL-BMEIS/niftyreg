#include "CPUOptimiseKernel.h"

CPUOptimiseKernel::CPUOptimiseKernel(Content *con, std::string name) : OptimiseKernel(name) {
    transformationMatrix = con->getTransformationMatrix();
    blockMatchingParams = con->getBlockMatchingParams();
}

void CPUOptimiseKernel::calculate(bool affine, bool ils, bool svd) {
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
