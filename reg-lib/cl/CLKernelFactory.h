#pragma once
#include "KernelFactory.h"


class Platform;
class KernelImpl;
class Context;
class CLKernelFactory : public KernelFactory
{
public:
	KernelImpl* createKernelImpl(std::string name, const Platform& platform, Context* con) const;
};

