#pragma once

#include "Kernel.h"
#include "Content.h"

class KernelFactory {
public:
    virtual Kernel* Produce(std::string name, Content *con) const = 0;
    virtual ~KernelFactory() {}
};
