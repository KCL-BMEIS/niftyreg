#ifndef KERNELFACTORY_H_
#define KERNELFACTORY_H_

#include "AladinContent.h"

class  KernelFactory {
public:

    virtual Kernel* produceKernel(std::string name, AladinContent* con) const = 0;
    virtual ~KernelFactory() {
    }
};



#endif /*KERNELFACTORY_H_*/
