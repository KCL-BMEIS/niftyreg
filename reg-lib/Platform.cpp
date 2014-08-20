

#include "Platform.h"
#include "Context.h"
#include <iostream>
#include "Kernel.h"
#include "KernelFactory.h"

using namespace std;



Platform::Platform() {

}
Platform::~Platform() {
}
Kernel Platform::createKernel(const string& name, unsigned int dType) const {
	return Kernel(kernelFactories.find(name)->second->createKernelImpl(name, *this, dType));
}
void Platform::shout() {
	std::cout<<"Helo from Platform" << std::endl;

	//exit(1);

}
void Platform::registerKernelFactory(const string& name, KernelFactory* factory) {
	kernelFactories[name] = factory;
}


