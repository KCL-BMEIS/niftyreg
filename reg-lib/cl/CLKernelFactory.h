#pragma once
#include "KernelFactory.h"


class Platform;
class KernelImpl;
class Context;
class CLKernelFactory : public KernelFactory
{
public:
	Kernel* createKernel(std::string name, Context* con) const;
};

