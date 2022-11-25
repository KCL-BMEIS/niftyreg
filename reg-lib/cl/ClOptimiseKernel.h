#pragma once

#include "OptimiseKernel.h"
#include "CLAladinContent.h"

class ClOptimiseKernel: public OptimiseKernel {
public:
    ClOptimiseKernel(Content *con);
    ~ClOptimiseKernel() {}
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam * blockMatchingParams;
    mat44 *transformationMatrix;
};
