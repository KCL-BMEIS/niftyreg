#ifndef CPUKERNLFACTORY_H
#define CPUKERNLFACTORY_H

#include "KernelFactory.h"

class AladinContent;

class CPUKernelFactory : public KernelFactory
{
public:
   Kernel *produceKernel(std::string name,  AladinContent *con) const;
};

#endif
