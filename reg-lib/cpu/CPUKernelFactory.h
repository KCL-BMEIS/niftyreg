#ifndef CPUKERNLFACTORY_H
#define CPUKERNLFACTORY_H

#include "KernelFactory.h"
#include "Context.h"


class CPUKernelFactory : public KernelFactory
{
public:
   Kernel* createKernel(std::string name,  Content* con) const;
};

#endif
