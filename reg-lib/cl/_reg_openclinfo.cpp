#include <iostream>
#include "InfoDevice.h"

void showCLInfo()
{
   ClContextSingleton *sContext = &ClContextSingleton::GetInstance();
   cl_uint numPlatforms = sContext->GetNumPlatforms();

   for (cl_uint i = 0; i < numPlatforms; i++)
   {
      cl_uint numDevices = sContext->GetNumDevices();
      cl_device_id * devices = sContext->GetDevices();
      NR_COUT << "-----------------------------------" << std::endl;
      NR_COUT << "[NiftyReg OPENCL] " << numDevices << "device(s) detected" << std::endl;
      NR_COUT << "-----------------------------------" << std::endl;
      // Iterate through each device, displaying associated information
      for (cl_uint j = 0; j < numDevices; j++)
      {
         NR_COUT << "[NiftyReg OPENCL] Device id " << j << std::endl;
         DeviceLog<char >::show(devices[j], CL_DEVICE_NAME, "Device Name");
//         DeviceLog<char >::show(devices[j], CL_DEVICE_VENDOR, "**** CL_DEVICE_VENDOR");
//         DeviceLog<char >::show(devices[j], CL_DRIVER_VERSION, "**** CL_DRIVER_VERSION");
         DeviceLog<char >::show(devices[j], CL_DEVICE_VERSION, "OpenCL version");
         DeviceLog<long long unsigned>::show(devices[j], CL_DEVICE_TYPE, "Device type");
         DeviceLog<unsigned>::show(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, "Multiprocessor number");
//         DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
         DeviceLog<size_t>::showKernelInfo(devices[j], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
//         DeviceLog<size_t> ::show(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
//         DeviceLog<size_t>::show(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
         DeviceLog<unsigned>::show(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, "Clock rate (Mhz)");
         DeviceLog<long long unsigned>::show(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, "Global memory size");
//         DeviceLog<cl_ulong>::show(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
//         DeviceLog<cl_uint>::show(devices[j], CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
//         DeviceLog<cl_device_local_mem_type>::show(devices[j], CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE");
         DeviceLog<long long unsigned>::show(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, "Device memory size");
//         DeviceLog<cl_bool>::show(devices[j], CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
//         DeviceLog<cl_bool>::show(devices[j], CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");
//         DeviceLog<cl_device_exec_capabilities>::show(devices[j], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");
//         DeviceLog<cl_command_queue_properties>::show(devices[j], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");
           DeviceLog<int>::show(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "Device double config");
           DeviceLog<int>::show(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, "Device double config");
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
         DeviceLog<int>::show(devices[j], CL_DEVICE_DOUBLE_FP_CONFIG, "Device double config");
#else
         DeviceLog<int>::show(devices[j], CL_DEVICE_SINGLE_FP_CONFIG, "Device single config only");
#endif
         NR_COUT << "-----------------------------------" << std::endl;
      }
   }
}
