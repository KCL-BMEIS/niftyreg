#pragma once
#include "KernelFactory.h"
#include "Content.h"

class Platform;
class Kernel;
class CudaKernelFactory : public KernelFactory
{
public:
	Kernel* produceKernel(std::string name, Content* con) const;
};

