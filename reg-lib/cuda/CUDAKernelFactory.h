#pragma once
#include "KernelFactory.h"
#include "Content.h"

class CUDAKernelFactory : public KernelFactory
{
public:
	Kernel *produceKernel(std::string name, Content *con) const;
};

