#pragma once

#include "KernelFactory.h"
#include "AladinContent.h"

class CudaKernelFactory: public KernelFactory {
public:
	Kernel* ProduceKernel(std::string name, AladinContent *con) const;
};
