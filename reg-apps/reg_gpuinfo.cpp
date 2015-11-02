#include "../reg-lib/_reg_gpuinfo.h"
#include "_reg_tools.h"

void showCPUInfo()
{
    reg_print_msg_error("You asked for CPU info not GPU info");
}

#ifdef _USE_CUDA
void showCUDAInfo(void)
{
    showCUDACardInfo();
}
#endif

#ifdef _USE_OPENCL
void showCLInfo(void)
{

    //	cl_int errNum;
    //	cl_platform_id * platformIds = sContext->getPlatformIds();
    CLContextSingletton *sContext = &CLContextSingletton::Instance();
    cl_uint numPlatforms = sContext->getNumPlatforms();

    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        cl_uint numDevices = sContext->getNumDevices();
        cl_device_id * devices = sContext->getDevices();
        std::cout <<std::endl<<"\t"<< "******************************************************" << std::endl;
        std::cout << "\t**** Number of devices: \t" << numDevices << std::endl;
        std::cout <<"\t"<< "******************************************************" << std::endl;
        // Iterate through each device, displaying associated information
        for (cl_uint j = 0; j < numDevices; j++)
        {
            std::cout <<std::endl<<"\t"<< "************************************************************************************" << std::endl;
            std::cout << "\t"<<"----------------------------------Device id: " << j << "--------------------------------------" << std::endl;
            DeviceLog<char >::show(devices[j], CL_DEVICE_NAME, "**** CL_DEVICE_NAME");
            DeviceLog<char >::show(devices[j], CL_DEVICE_VENDOR, "**** CL_DEVICE_VENDOR");
            DeviceLog<char >::show(devices[j], CL_DRIVER_VERSION, "**** CL_DRIVER_VERSION");
            DeviceLog<char >::show(devices[j], CL_DEVICE_VERSION, "**** CL_DEVICE_VERSION");
            std::cout <<"\t"<< "************************************************************************************" << std::endl<< std::endl;

            DeviceLog<cl_device_type>::show(devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
            DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
            DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
            DeviceLog<size_t>::showKernelInfo(devices[j], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
            DeviceLog<size_t> ::show(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
            DeviceLog<size_t>::show(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
            DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
            DeviceLog<cl_ulong>::show(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
            DeviceLog<cl_ulong>::show(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
            DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
            DeviceLog<cl_device_local_mem_type>::show(devices[j], CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE");
            DeviceLog<cl_ulong>::show(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
            DeviceLog<cl_bool>::show(devices[j], CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
            DeviceLog<cl_bool>::show(devices[j], CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");
            DeviceLog<cl_device_exec_capabilities>::show(devices[j], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");
            DeviceLog<cl_command_queue_properties>::show(devices[j], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");
            DeviceLog<cl_int>::show(devices[j], CL_DEVICE_DOUBLE_FP_CONFIG, "CL_DEVICE_DOUBLE_FP_CONFIG");
            std::cout << std::endl;
        }
    }
}
#endif
/* *************************************************************** */
int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int platformCode = atoi(argv[1]);
    if (platformCode == NR_PLATFORM_CPU) {
        showCPUInfo();
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        showCUDAInfo();
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        showCLInfo();
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
