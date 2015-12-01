#pragma once
#include "KernelFactory.h"
#include "AladinContent.h"

class CUDAKernelFactory : public KernelFactory
{
public:
	Kernel *produceKernel(std::string name, AladinContent *con) const;
};

