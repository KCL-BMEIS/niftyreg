#ifndef CLOPTIMISEKERNEL_H
#define CLOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "CLAladinContent.h"

class CLOptimiseKernel : public OptimiseKernel
{
    public:

       CLOptimiseKernel(AladinContent * con, std::string name);
       ~CLOptimiseKernel();
       void calculate(bool affine);
    private:
       _reg_blockMatchingParam * blockMatchingParams;
       mat44 *transformationMatrix;
       CLContextSingletton *sContext;
       ClAladinContent  *con;
};

#endif // CLOPTIMISEKERNEL_H
