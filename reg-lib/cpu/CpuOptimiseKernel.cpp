#include "CpuOptimiseKernel.h"

CpuOptimiseKernel::CpuOptimiseKernel(AladinContent *con, std::string name) : OptimiseKernel(name) {
    transformationMatrix = con->GetTransformationMatrix();
    blockMatchingParams = con->GetBlockMatchingParams();
}

void CpuOptimiseKernel::Calculate(bool affine) {
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
