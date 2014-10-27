#pragma once
#include "KernelFactory.h"
#include "Context.h"

class Platform;
class KernelImpl;
class CpuContext;

class CPUKernelFactory : public KernelFactory
{
public:
	Kernel* createKernel(std::string name,  Context* con) const;
};

