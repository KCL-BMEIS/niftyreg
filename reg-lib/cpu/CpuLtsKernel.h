#pragma once

#include "LtsKernel.h"
#include "AladinContent.h"

class CpuLtsKernel: public LtsKernel {
public:
    CpuLtsKernel(Content *con);
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
};
