#ifndef KERNELFACTORY_H_
#define KERNELFACTORY_H_


#include "KernelImpl.h"
#include "Context.h"

/**
 * A KernelFactory is an object that can create KernelImpls.  This is an abstract class.
 * Each Platform maintains a list of KernelFactory objects that it uses to create
 * KernelImpls as needed.
 */

class  KernelFactory {
public:
    /**
     * Create a KernelImpl.
     * 
     * @param name the name of the kernel to create
     * @param context the context the kernel will belong to
     */
    virtual KernelImpl* createKernelImpl(std::string name, const Platform& platform, Context* con) const = 0;
    virtual ~KernelFactory() {
    }
};



#endif /*KERNELFACTORY_H_*/
