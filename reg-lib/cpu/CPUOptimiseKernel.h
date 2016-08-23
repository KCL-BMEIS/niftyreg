#ifndef CPUOPTIMISEKERNEL_H
#define CPUOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "_reg_blockMatching.h"
#include "nifti1_io.h"
#include "AladinContent.h"

class CPUOptimiseKernel : public OptimiseKernel {
public:
    CPUOptimiseKernel(GlobalContent *con, std::string name);
    void calculate(bool affine);

private:
    AladinContent* con;
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;

};

#endif // CPUOPTIMISEKERNEL_H
