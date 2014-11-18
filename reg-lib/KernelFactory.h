#ifndef KERNELFACTORY_H_
#define KERNELFACTORY_H_

#include "Context.h"

class  KernelFactory {
public:

    virtual Kernel* createKernel(std::string name, Context* con) const = 0;
    virtual ~KernelFactory() {
    }
};



#endif /*KERNELFACTORY_H_*/
