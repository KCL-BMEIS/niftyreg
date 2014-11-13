#ifndef PLATFORM_H_
#define PLATFORM_H_


#include <map>
#include <string>
#include <vector>

class Kernel;
class KernelFactory;
class Context;


class  Platform {
public:
	Platform();
	void shout();
	Kernel* createKernel(const std::string& name, Context* con) const;
	void registerKernelFactory(const std::string& name, KernelFactory* factory);

	std::map<std::string, KernelFactory*> kernelFactories;
	virtual std::string getName(){ return ""; }

	virtual ~Platform();

};



#endif //PLATFORM_H_
