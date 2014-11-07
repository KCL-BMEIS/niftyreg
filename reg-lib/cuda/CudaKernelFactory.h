#pragma once
#include "KernelFactory.h"
#include "Context.h"

class Platform;
class Kernel;
class CudaKernelFactory : public KernelFactory
{
public:
	Kernel* createKernel(std::string name, Context* con) const;
};

