#ifndef CPUPLATFORM_H_
#define CPUPLATFORM_H_

#include "Context.h"
#include "Platform.h"

class CPUPlatform: public Platform
{
public:
	CPUPlatform();

	//set platform specific data to context
	void setPlatformData(Context &ctx);
	std::string getName(){ return "cpu_platform"; }

private:
	unsigned int getRank();
};
#endif //CPUPLATFORM_H_