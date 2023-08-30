#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "ClContextSingleton.h"

template<typename T>
class DeviceLog {
public:

	static void appendToString(bool flag, std::string name, std::string & str)
	{
		if (flag) {
			if(str.length() > 0)  str.append(" / ") ;
			str.append(name);
		}
	}

	static void show(cl_device_id id, cl_device_info name, std::string str)
	{
		std::size_t paramValueSize;
		std::string clInfo;
		ClContextSingleton *sContext = &ClContextSingleton::GetInstance();

		sContext->CheckErrNum(clGetDeviceInfo(id, name, 0, nullptr, &paramValueSize), "Failed to find OpenCL device info ");

		T * field = (T *) alloca(sizeof(T) * paramValueSize);
		sContext->CheckErrNum(clGetDeviceInfo(id, name, paramValueSize, field, nullptr), "Failed to find OpenCL device info ");

		switch (name) {
		case CL_DEVICE_TYPE: {
				const cl_device_type deviceType = *(reinterpret_cast<cl_device_type*>(field));
				appendToString(deviceType & CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", clInfo);
				appendToString(deviceType & CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", clInfo);
				appendToString(deviceType & CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", clInfo);
				appendToString(deviceType & CL_DEVICE_TYPE_DEFAULT, "CL_DEVICE_TYPE_DEFAULT", clInfo);
				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << clInfo << std::endl;
			}
			break;
		case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: {
				const cl_device_mem_cache_type cacheType = *(reinterpret_cast<cl_device_mem_cache_type*>(field));
				appendToString(cacheType & CL_NONE, "CL_NONE", clInfo);
				appendToString(cacheType & CL_READ_ONLY_CACHE, "CL_READ_ONLY_CACHE", clInfo);
				appendToString(cacheType & CL_READ_WRITE_CACHE, "CL_READ_WRITE_CACHE", clInfo);

				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << clInfo << std::endl;
			}
			break;
		case CL_DEVICE_LOCAL_MEM_TYPE: {
				const cl_device_local_mem_type localMemType = *(reinterpret_cast<cl_device_local_mem_type*>(field));
				appendToString(localMemType & CL_LOCAL, "CL_LOCAL", clInfo);
				appendToString(localMemType & CL_GLOBAL, "CL_GLOBAL", clInfo);

				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << clInfo << std::endl;
			}
			break;
		case CL_DEVICE_EXECUTION_CAPABILITIES: {

				const cl_device_exec_capabilities execCapabilities = *(reinterpret_cast<cl_device_exec_capabilities*>(field));

				appendToString(execCapabilities & CL_EXEC_KERNEL, "CL_EXEC_KERNEL", clInfo);
				appendToString(execCapabilities & CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL", clInfo);

				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << clInfo << std::endl;
			}
			break;
		case CL_DEVICE_QUEUE_PROPERTIES: {

				appendToString(*(reinterpret_cast<cl_device_exec_capabilities*>(field)) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", clInfo);
				appendToString(*(reinterpret_cast<cl_device_exec_capabilities*>(field)) & CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", clInfo);

				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << clInfo << std::endl;
			}
			break;
		case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
				cl_uint maxWorkItemDimensions;

				sContext->CheckErrNum(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, nullptr), "Failed to find OpenCL device info  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS.");
				NR_COUT << str << ":\t";
				for (cl_uint i = 0; i < maxWorkItemDimensions; i++)
					NR_COUT << field[i] << " ";
				NR_COUT << std::endl;
			}
			break;

		case CL_DEVICE_NAME:
		case CL_DEVICE_VENDOR:
		case CL_DRIVER_VERSION:
		case CL_DEVICE_VERSION: {
				NR_COUT << "[NiftyReg OPENCL] " << str << ": " << field << std::endl;
			}
			break;
		default:
			NR_COUT << "[NiftyReg OPENCL] " << str << ": " << *field << std::endl;
			break;
		}
	}
	static void showKernelInfo(cl_device_id id, cl_kernel_work_group_info name, std::string str)
	{
		cl_int errNum;
		size_t local;
		ClContextSingleton *sContext = &ClContextSingleton::GetInstance();

		errNum = clGetKernelWorkGroupInfo(sContext->DummyKernel(id), id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, nullptr);

		switch (name) {
		case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
				if (errNum != CL_SUCCESS)  local = 1;
				NR_COUT << "[NiftyReg OPENCL] Warp / wavefront" << ": " << local << std::endl;
			}
			break;
			break;
		default:
			NR_COUT << "[NiftyReg OPENCL] " << str << ": " << local << std::endl;
			break;
		}
	}
};
