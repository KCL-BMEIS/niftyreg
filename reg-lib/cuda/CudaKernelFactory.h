#pragma once

#include "KernelFactory.h"

class CudaKernelFactory: public KernelFactory {
public:
	Kernel* ProduceKernel(std::string name, Content *con) const;
};
