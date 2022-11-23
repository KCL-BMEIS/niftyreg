#pragma once

#include "AladinContent.h"

class KernelFactory {
public:
    virtual Kernel* ProduceKernel(std::string name, AladinContent* con) const = 0;
    virtual ~KernelFactory() {}
};
