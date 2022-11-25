#pragma once

#include "KernelFactory.h"

class CpuKernelFactory: public KernelFactory {
public:
   Kernel* ProduceKernel(std::string name, Content *con) const;
};
