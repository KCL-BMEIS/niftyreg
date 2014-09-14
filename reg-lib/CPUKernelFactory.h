#pragma once
#include "KernelFactory.h"
#include "Context.h"

class Platform;
class KernelImpl;
class CPUKernelFactory : public KernelFactory
{
public:
	KernelImpl* createKernelImpl(std::string name, const Platform& platform, Context* con) const;
};

