#pragma once

#include "OptimiseKernel.h"
#include "_reg_blockMatching.h"
#include "niftilib/nifti1_io.h"
#include "AladinContent.h"

class CpuOptimiseKernel: public OptimiseKernel {
public:
    CpuOptimiseKernel(Content *con);
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
};
