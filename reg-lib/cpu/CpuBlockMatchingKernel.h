#pragma once

#include "BlockMatchingKernel.h"
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
