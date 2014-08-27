#pragma once

#include "Context.h"
#include "Platform.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class CLPlatform : public Platform
{
public:
	CLPlatform();


	//try this later
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_device_id device = 0;

	//set platform specific data to context
	void setPlatformData(Context &ctx);

private:
	unsigned int getRank();
};
