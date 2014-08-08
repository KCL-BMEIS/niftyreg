#ifndef KERNEL_H_
#define KERNEL_H_

#include "KernelImpl.h"



/**
 * A Kernel encapsulates a particular implementation of a calculation that can be performed on the data
 * in a Context.  Kernel objects are created by Platforms:
 * 
 * <pre>
 * Kernel kernel = platform.createKernel(kernelName);
 * </pre>
 * 
 * The Kernel class itself does not specify any details of what calculation is to be performed or the API
 * for calling it.  Instead, subclasses of KernelImpl will define APIs which are appropriate to particular
 * calculations.  To execute a Kernel, you therefore request its implementation object and cast it to the
 * correct type:
 * 
 * <pre>
 * dynamic_cast<AddStreamsImpl&>(kernel.getImpl()).execute(stream1, stream2);
 * </pre>
 */

class Kernel {
public:
    Kernel();
    Kernel(const Kernel& copy);
    /**
     * Create a Kernel that wraps a KernelImpl.
     * 
     * @param name the name of the kernel to create
     */
    Kernel(KernelImpl* impl);
    ~Kernel();
    Kernel& operator=(const Kernel& copy);
    /**
     * Get the name of this Kernel.
     */
    std::string getName() const;
    /**
     * Get the object which implements this Kernel.
     */
    const KernelImpl& getImpl() const;
    /**
     * Get the object which implements this Kernel.
     */
    KernelImpl& getImpl();
    /**
     * Get a reference to the object which implements this Kernel, casting it to the specified type.
     */
    template <class T>
    const T& getAs() const {
        return dynamic_cast<T&>(*impl);
    }
    /**
     * Get a reference to the object which implements this Kernel, casting it to the specified type.
     */
    template <class T>
    T& getAs() {
        return dynamic_cast<T&>(*impl);
    }
private:
    KernelImpl* impl;
};



#endif /*KERNEL_H_*/
