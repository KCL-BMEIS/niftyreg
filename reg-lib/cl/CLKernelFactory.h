#ifndef CLPKERNELFACTORY_H
#define CLPKERNELFACTORY_H

#include "KernelFactory.h"

class Content;
class CLKernelFactory : public KernelFactory
{
public:
   Kernel* createKernel(std::string name, Content* con) const;
};

#endif
