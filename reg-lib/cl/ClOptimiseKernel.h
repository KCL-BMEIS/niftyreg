#pragma once

#include "OptimiseKernel.h"
#include "CLAladinContent.h"

class ClOptimiseKernel : public OptimiseKernel
{
    public:

       ClOptimiseKernel(AladinContent * con, std::string name);
       ~ClOptimiseKernel();
       void Calculate(bool affine);
    private:
       _reg_blockMatchingParam * blockMatchingParams;
       mat44 *transformationMatrix;
       ClContextSingleton *sContext;
       ClAladinContent  *con;
};
