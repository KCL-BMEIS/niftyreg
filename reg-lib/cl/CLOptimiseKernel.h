#ifndef CLOPTIMISEKERNEL_H
#define CLOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "CLContent.h"

class CLOptimiseKernel : public OptimiseKernel
{
    public:

       CLOptimiseKernel(Content * con, std::string name);
       ~CLOptimiseKernel();
       void calculate(bool affine, bool ils = 0);
    private:
       _reg_blockMatchingParam * blockMatchingParams;
       mat44 *transformationMatrix;
       CLContextSingletton *sContext;
       ClContent  *con;
};

#endif // CLOPTIMISEKERNEL_H
