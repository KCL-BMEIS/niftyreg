#ifndef PLATFORM_H_
#define PLATFORM_H_

#include <map>
#include <string>
#include <vector>

#define NR_PLATFORM_CPU 0
#define NR_PLATFORM_CUDA 1
#define NR_PLATFORM_CL 2

class Kernel;
class KernelFactory;
class Content;

class  Platform {
public:
	Platform();
	Kernel* createKernel(const std::string& name, Content* con) const;
	void assignKernelToFactory(const std::string& name, KernelFactory* factory);

	std::map<std::string, KernelFactory*> kernelFactories;
	virtual std::string getName(){ return ""; }

	virtual ~Platform();

};



#endif //PLATFORM_H_
