#pragma once

#include "Kernel.h"

class BlockMatchingKernel : public Kernel {
public:
    static std::string GetName() {
        return "blockMatchingKernel";
    }
    BlockMatchingKernel(std::string name) : Kernel(name) {

    }
    virtual ~BlockMatchingKernel(){}
    virtual void Calculate() = 0;
};
