#pragma once

#include "BlockMatchingKernel.h"
#include "AladinContent.h"

class CpuBlockMatchingKernel: public BlockMatchingKernel {
public:
    CpuBlockMatchingKernel(Content *con);
    virtual void Calculate() override;

private:
    nifti_image *reference;
    nifti_image *warped;
    _reg_blockMatchingParam* params;
    int *mask;
};
