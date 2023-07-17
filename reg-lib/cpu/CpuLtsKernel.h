#pragma once

#include "LtsKernel.h"
#include "_reg_blockMatching.h"
#include "niftilib/nifti1_io.h"
#include "AladinContent.h"

class CpuLtsKernel: public LtsKernel {
public:
    CpuLtsKernel(Content *con);
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
};
