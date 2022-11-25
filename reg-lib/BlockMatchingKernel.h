#pragma once

#include "Kernel.h"

class BlockMatchingKernel: public Kernel {
public:
    static std::string GetName() {
        return "BlockMatchingKernel";
    }
    BlockMatchingKernel() : Kernel() {}
    virtual ~BlockMatchingKernel() {}
    virtual void Calculate() = 0;
};
