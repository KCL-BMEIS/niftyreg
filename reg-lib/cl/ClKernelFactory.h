#pragma once

#include "KernelFactory.h"
#include "AladinContent.h"

class ClKernelFactory: public KernelFactory {
public:
   Kernel* ProduceKernel(std::string name, AladinContent *con) const;
};
