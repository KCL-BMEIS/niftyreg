#ifndef BLOCKMATCHINGKERNEL_H
#define BLOCKMATCHINGKERNEL_H

#include "Kernel.h"

class BlockMatchingKernel : public Kernel {
public:
    static std::string getName() {
        return "blockMatchingKernel";
    }
    BlockMatchingKernel(std::string name) : Kernel(name) {

    }
    virtual ~BlockMatchingKernel(){}
    virtual void calculate() = 0;
};

#endif // BLOCKMATCHINGKERNEL_H
