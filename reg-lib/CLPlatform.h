#pragma once

#include "Context.h"
#include "Platform.h"

class CLPlatform : public Platform
{
public:
	CLPlatform();

	//set platform specific data to context
	void setPlatformData(Context &ctx);

private:
	unsigned int getRank();
};
