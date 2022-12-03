#pragma once

#include "KernelFactory.h"

class CpuKernelFactory: public KernelFactory {
public:
   Kernel* Produce(std::string name, Content *con) const;
};
