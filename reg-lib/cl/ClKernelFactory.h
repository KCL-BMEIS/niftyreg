#pragma once

#include "KernelFactory.h"

class ClKernelFactory: public KernelFactory {
public:
   Kernel* Produce(std::string name, Content *con) const;
};
