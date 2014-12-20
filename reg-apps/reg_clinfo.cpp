#include <iostream>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <malloc.h> // needed for alloca
#endif // _WIN32
#if defined(linux) || defined(__APPLE__) || defined(__MACOSX)
# include <alloca.h>
#endif // linux

#include "../reg-lib/cl/CLContextSingletton.h"
#include "../reg-lib/cl/InfoDevice.h"

void displayInfo(void) {
	cl_int errNum;
	CLContextSingletton *sContext = &CLContextSingletton::Instance();
	cl_uint numPlatforms = sContext->getNumPlatforms();
	cl_platform_id * platformIds = sContext->getPlatformIds();


	for (cl_uint i = 0; i < numPlatforms; i++) {

		cl_uint numDevices = sContext->getNumDevices();
		cl_device_id * devices = sContext->getDevices();

		std::cout << "\tNumber of devices: \t" << numDevices << std::endl;
		// Iterate through each device, displaying associated information
		for (cl_uint j = 0; j < numDevices; j++) {
			std::cout << "============================================================================================" << std::endl;
			std::cout << "----------------------------------Device id: " << j << "----------------------------------------------" << std::endl;
			std::cout << "============================================================================================" << std::endl;
			std::cout <<std::endl<<"\t"<< "******************************************************" << std::endl;

			InfoDevice<ArrayType<char> >::display(devices[j], CL_DEVICE_NAME, "**** CL_DEVICE_NAME");

			InfoDevice<ArrayType<char> >::display(devices[j], CL_DEVICE_VENDOR, "**** CL_DEVICE_VENDOR");

			InfoDevice<ArrayType<char> >::display(devices[j], CL_DRIVER_VERSION, "**** CL_DRIVER_VERSION");

			InfoDevice<ArrayType<char> >::display(devices[j], CL_DEVICE_VERSION, "**** CL_DEVICE_VERSION");
			std::cout <<"\t"<< "******************************************************" << std::endl<<std::endl;

			InfoDevice<cl_device_type>::display(devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");

			InfoDevice<cl_uint>::display(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");

			InfoDevice<cl_uint>::display(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");

			InfoDevice<ArrayType<size_t> >::display(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");

			InfoDevice<std::size_t>::display(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");

			InfoDevice<cl_uint>::display(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");

			InfoDevice<cl_ulong>::display(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");

			InfoDevice<cl_ulong>::display(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");

			InfoDevice<cl_uint>::display(devices[j], CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");

			InfoDevice<cl_device_local_mem_type>::display(devices[j], CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE");

			InfoDevice<cl_ulong>::display(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");

			InfoDevice<cl_bool>::display(devices[j], CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");

			InfoDevice<cl_bool>::display(devices[j], CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");

			InfoDevice<cl_device_exec_capabilities>::display(devices[j], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");

			InfoDevice<cl_command_queue_properties>::display(devices[j], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");

			std::cout << std::endl;
		}
	}
}


int main(int argc, char** argv) {


	displayInfo();

	return 0;
}
