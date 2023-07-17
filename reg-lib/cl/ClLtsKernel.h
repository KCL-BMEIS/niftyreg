#pragma once

#include "LtsKernel.h"
#include "ClAladinContent.h"

class ClLtsKernel: public LtsKernel {
public:
    ClLtsKernel(Content *con);
    ~ClLtsKernel() {}
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam * blockMatchingParams;
    mat44 *transformationMatrix;
};
