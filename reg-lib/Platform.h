#ifndef PLATFORM_H_
#define PLATFORM_H_


#include <map>
#include <string>
#include <vector>

class Kernel;
class KernelFactory;
class Context;

/**
 * A Platform defines an implementation of all the kernels needed to perform some calculation.
 * More precisely, a Platform object acts as a registry for a set of KernelFactory
 * objects which together implement the kernels.  The Platform class, in turn, provides a
 * static registry of all available Platform objects.
 * 
 * To get a Platform object, call
 * 
 * <pre>
 * Platform& platform Platform::findPlatform(kernelNames);
 * </pre>
 * 
 * passing in the names of all kernels that will be required for the calculation you plan to perform.  It
 * will return the fastest available Platform which provides implementations of all the specified kernels.
 * You can then call createKernel() to construct particular kernels as needed.
 */

class  Platform {
public:
	Platform();
	void shout();
	Kernel createKernel(const std::string& name, Context* con) const;
	void registerKernelFactory(const std::string& name, KernelFactory* factory);

	std::map<std::string, KernelFactory*> kernelFactories;
	virtual ~Platform();

};



#endif PLATFORM_H_*/
