#pragma once

#include "LtsKernel.h"
#include "ClAladinContent.h"

class ClLtsKernel: public LtsKernel {
public:
    ClLtsKernel(Content *con);
    virtual void Calculate(bool affine) override;

private:
    ClAladinContent *con;
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
};
