#include "Platform.h"
#include "Context.h"
#include "KernelFactory.h"

using namespace std;



Platform::Platform()
{
}

Platform::~Platform()
{
}

Kernel* Platform::createKernel(const string& name, Context* con) const {
	return kernelFactories.find(name)->second->createKernel(name, con);
}
void Platform::registerKernelFactory(const string& name, KernelFactory* factory) {
	kernelFactories[name] = factory;
}


