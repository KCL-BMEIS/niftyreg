#ifndef CPUKERNLFACTORY_H
#define CPUKERNLFACTORY_H

#include "KernelFactory.h"


class CPUKernelFactory : public KernelFactory
{
public:
   Kernel *produceKernel(std::string name,  GlobalContent *con) const;
};

#endif
