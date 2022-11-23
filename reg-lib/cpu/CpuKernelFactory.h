#pragma once

#include "KernelFactory.h"

class AladinContent;

class CpuKernelFactory: public KernelFactory {
public:
   Kernel* ProduceKernel(std::string name, AladinContent *con) const;
};
