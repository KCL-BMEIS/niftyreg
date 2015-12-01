#ifndef CPUBLOCKMATCHINGKERNEL_H
#define CPUBLOCKMATCHINGKERNEL_H

#include "BlockMatchingKernel.h"
#include "_reg_blockMatching.h"
#include "nifti1_io.h"
#include "AladinContent.h"

class CPUBlockMatchingKernel : public BlockMatchingKernel {
public:

    CPUBlockMatchingKernel(AladinContent *con, std::string name);

    void calculate();

    nifti_image *reference;
    nifti_image *warped;
    _reg_blockMatchingParam* params;
    int *mask;

};

#endif // CPUBLOCKMATCHINGKERNEL_H
