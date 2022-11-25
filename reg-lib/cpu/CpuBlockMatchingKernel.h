#pragma once

#include "BlockMatchingKernel.h"
#include "_reg_blockMatching.h"
#include "nifti1_io.h"
#include "AladinContent.h"

class CpuBlockMatchingKernel: public BlockMatchingKernel {
public:
    CpuBlockMatchingKernel(Content *con);
    void Calculate();

private:
    nifti_image *reference;
    nifti_image *warped;
    _reg_blockMatchingParam* params;
    int *mask;
};
