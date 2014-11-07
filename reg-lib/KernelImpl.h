#ifndef KERNELIMPL_H_
#define KERNELIMPL_H_



#include "Platform.h"
#include <string>
#include <cassert>




/**
 * A KernelImpl defines the internal implementation of a Kernel object.  A subclass will typically
 * declare an abstract execute() method which defines the API for executing the kernel.  Other classes
 * will in turn subclass it and provide concrete implementations of the execute() method.
 */

class KernelImpl {
public:

	KernelImpl(std::string nameIn, const Platform& platform);
	virtual ~KernelImpl() {
		assert(referenceCount == 0);
		
	}
	std::string getName() const;
	const Platform& getPlatform();
	unsigned int referenceCount;
	std::string name;
	const Platform *platform;


};



#endif /*KERNELIMPL_H_*/
