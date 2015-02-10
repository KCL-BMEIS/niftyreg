#ifndef CPUKERNLFACTORY_H
#define CPUKERNLFACTORY_H

#include "KernelFactory.h"

class Content;

class CPUKernelFactory : public KernelFactory
{
public:
   Kernel *produceKernel(std::string name,  Content *con) const;
};

#endif
