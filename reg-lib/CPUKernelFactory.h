#pragma once
#include "KernelFactory.h"
#include "Context.h"

class Platform;
class KernelImpl;
class CPUKernelFactory : public KernelFactory
{
public:
	KernelImpl* createKernelImpl(std::string name, const Platform& platform, unsigned int dType) const;
	template<class T> KernelImpl* registerDataTypes(std::string name, const Platform& platform) const;
};

