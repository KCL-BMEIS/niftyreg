#pragma once

#include "LtsKernel.h"
#include "CudaAladinContent.h"

// Kernel functions for numerical optimisation
class CudaLtsKernel: public LtsKernel {
public:
    CudaLtsKernel(Content *conIn);
    virtual void Calculate(bool affine) override;

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
    CudaAladinContent *con;
};
