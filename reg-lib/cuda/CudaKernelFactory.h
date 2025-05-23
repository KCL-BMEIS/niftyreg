#pragma once

#include "KernelFactory.h"

class CudaKernelFactory: public KernelFactory {
public:
	Kernel* Produce(std::string name, Content *con) const;
};
