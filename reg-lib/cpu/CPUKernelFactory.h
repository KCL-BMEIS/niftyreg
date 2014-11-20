#pragma once
#include "KernelFactory.h"
#include "Context.h"


class CPUKernelFactory : public KernelFactory
{
public:
	Kernel* createKernel(std::string name,  Context* con) const;
};

