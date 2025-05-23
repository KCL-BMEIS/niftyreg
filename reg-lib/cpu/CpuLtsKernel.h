#pragma once

#include "LtsKernel.h"
#include "AladinContent.h"

class CpuLtsKernel: public LtsKernel {
public:
    CpuLtsKernel(Content *con);
    virtual void Calculate(bool affine) override;

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
};
