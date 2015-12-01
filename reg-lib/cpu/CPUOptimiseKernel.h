#ifndef CPUOPTIMISEKERNEL_H
#define CPUOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "_reg_blockMatching.h"
#include "nifti1_io.h"
#include "AladinContent.h"

class CPUOptimiseKernel : public OptimiseKernel {
public:
    CPUOptimiseKernel(AladinContent *con, std::string name);

    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;

    void calculate(bool affine);

};

#endif // CPUOPTIMISEKERNEL_H
