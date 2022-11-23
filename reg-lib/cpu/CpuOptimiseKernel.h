#pragma once

#include "OptimiseKernel.h"
#include "_reg_blockMatching.h"
#include "nifti1_io.h"
#include "AladinContent.h"

class CpuOptimiseKernel : public OptimiseKernel {
public:
    CpuOptimiseKernel(AladinContent *con, std::string name);

    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;

    void Calculate(bool affine);

};
