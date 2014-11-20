#ifndef PLATFORM_H_
#define PLATFORM_H_


#include <map>
#include <string>
#include <vector>

class Kernel;
class KernelFactory;
class Context;

#define NR_PLATFORM_CPU 0
#define NR_PLATFORM_CUDA 1
#define NR_PLATFORM_CL 2

class  Platform {
public:
	Platform();
	Kernel* createKernel(const std::string& name, Context* con) const;
	void registerKernelFactory(const std::string& name, KernelFactory* factory);

	std::map<std::string, KernelFactory*> kernelFactories;
	virtual std::string getName(){ return ""; }

	virtual ~Platform();

};



#endif //PLATFORM_H_
