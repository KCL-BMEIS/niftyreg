#ifndef CPUPLATFORM_H_
#define CPUPLATFORM_H_

#include "Context.h"
#include "Platform.h"

class CPUPlatform: public Platform
{
public:
	CPUPlatform();
	std::string getName(){ return "cpu_platform"; }

};
#endif //CPUPLATFORM_H_
