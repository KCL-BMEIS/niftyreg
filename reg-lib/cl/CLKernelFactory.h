#ifndef CLPKERNELFACTORY_H
#define CLPKERNELFACTORY_H

#include "KernelFactory.h"
#include "Content.h"

class CLKernelFactory : public KernelFactory
{
public:
   Kernel *produceKernel(std::string name, Content *con) const;
};

#endif
