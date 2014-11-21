#ifndef CLPLATFORM_H
#define CLPLATFORM_H

#include "Context.h"
#include "Platform.h"

class CLPlatform : public Platform
{
public:
	CLPlatform();
	std::string getName(){ return "cl_platform"; }

};
#endif
