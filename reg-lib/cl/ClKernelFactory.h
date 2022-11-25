#pragma once

#include "KernelFactory.h"

class ClKernelFactory: public KernelFactory {
public:
   Kernel* ProduceKernel(std::string name, Content *con) const;
};
