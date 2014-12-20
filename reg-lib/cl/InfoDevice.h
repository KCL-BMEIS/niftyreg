#include <iostream>
#include <fstream>
#include <sstream>

#ifndef INFODEVICE_H_
#define INFODEVICE_H_

template<typename T>
void appendBitfield(T info, T value, std::string name, std::string & str) {
	if (info & value) {
		if (str.length() > 0) {
			str.append(" | ");
		}
		str.append(name);
	}
}

template<typename T>
class InfoDevice {
public:
	static void display(cl_device_id id, cl_device_info name, std::string str) {
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo(id, name, 0, NULL, &paramValueSize);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *) alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo(id, name, paramValueSize, info, NULL);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		// Handle a few special cases
		switch (name) {
		case CL_DEVICE_TYPE: {
			std::string deviceType;

			appendBitfield<cl_device_type>(*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", deviceType);

			appendBitfield<cl_device_type>(*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", deviceType);

			appendBitfield<cl_device_type>(*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", deviceType);

			appendBitfield<cl_device_type>(*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_DEFAULT, "CL_DEVICE_TYPE_DEFAULT", deviceType);

			std::cout << "\t" << str << ":\t" << deviceType << std::endl;
		}
			break;
		case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: {
			std::string memType;

			appendBitfield<cl_device_mem_cache_type>(*(reinterpret_cast<cl_device_mem_cache_type*>(info)), CL_NONE, "CL_NONE", memType);
			appendBitfield<cl_device_mem_cache_type>(*(reinterpret_cast<cl_device_mem_cache_type*>(info)), CL_READ_ONLY_CACHE, "CL_READ_ONLY_CACHE", memType);

			appendBitfield<cl_device_mem_cache_type>(*(reinterpret_cast<cl_device_mem_cache_type*>(info)), CL_READ_WRITE_CACHE, "CL_READ_WRITE_CACHE", memType);

			std::cout << "\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_LOCAL_MEM_TYPE: {
			std::string memType;

			appendBitfield<cl_device_local_mem_type>(*(reinterpret_cast<cl_device_local_mem_type*>(info)), CL_GLOBAL, "CL_LOCAL", memType);

			appendBitfield<cl_device_local_mem_type>(*(reinterpret_cast<cl_device_local_mem_type*>(info)), CL_GLOBAL, "CL_GLOBAL", memType);

			std::cout << "\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_EXECUTION_CAPABILITIES: {
			std::string memType;

			appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_EXEC_KERNEL, "CL_EXEC_KERNEL", memType);

			appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL", memType);

			std::cout << "\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_QUEUE_PROPERTIES: {
			std::string memType;

			appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", memType);

			appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", memType);

			std::cout << "\t" << str << ":\t" << memType << std::endl;
		}
			break;
		default:
			std::cout << "\t" << str << ":\t" << *info << std::endl;
			break;
		}
	}
};

///
// Simple trait class used to wrap base types.
//
template<typename T>
class ArrayType {
public:
	static bool isChar() {
		return false;
	}
};

///
// Specialized for the char (i.e. null terminated string case).
//
template<>
class ArrayType<char> {
public:
	static bool isChar() {
		return true;
	}
};

///
// Specialized instance of class InfoDevice for array types.
//
template<typename T>
class InfoDevice<ArrayType<T> > {
public:
	static void display(cl_device_id id, cl_device_info name, std::string str) {
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo(id, name, 0, NULL, &paramValueSize);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *) alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo(id, name, paramValueSize, info, NULL);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		if (ArrayType<T>::isChar()) {
			std::cout << "\t" << str << ":\t" << info << std::endl;
		} else if (name == CL_DEVICE_MAX_WORK_ITEM_SIZES) {
			cl_uint maxWorkItemDimensions;

			errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL);
			if (errNum != CL_SUCCESS) {
				std::cerr << "Failed to find OpenCL device info " << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS." << std::endl;
				return;
			}

			std::cout << "\t" << str << ":\t";
			for (cl_uint i = 0; i < maxWorkItemDimensions; i++) {
				std::cout << info[i] << " ";
			}
			std::cout << std::endl;
		}
	}
};


#endif /* INFODEVICE_H_ */
