#ifndef CLPKERNELFACTORY_H
#define CLPKERNELFACTORY_H

#include "KernelFactory.h"
#include "AladinContent.h"

class CLKernelFactory : public KernelFactory
{
public:
   Kernel *produceKernel(std::string name, AladinContent *con) const;
};

#endif
