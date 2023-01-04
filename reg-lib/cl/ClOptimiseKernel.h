#pragma once

#include "OptimiseKernel.h"
#include "ClAladinContent.h"

class ClOptimiseKernel: public OptimiseKernel {
public:
    ClOptimiseKernel(Content *con);
    ~ClOptimiseKernel() {}
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam * blockMatchingParams;
    mat44 *transformationMatrix;
};
